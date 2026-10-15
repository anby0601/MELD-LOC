

#!/usr/bin/env python3
"""Trigger and record a CARLA lane-route run — v2 logger.

v2 adds THREE NDT-internal quantities that the 35-column v1 schema never captured. The 35 existing
columns keep their names and order, so every v1 parser (b2.load_route_rows, LA.load_signals) works
unchanged on v2 logs; the new columns are appended.

  ndt_nvtl        /localization/pose_estimator/nearest_voxel_transformation_likelihood
                  Autoware judges NDT convergence with NVTL, not transform_probability
                  (ndt_scan_matcher.param.yaml: converged_param_type: 1, threshold 2.3).
  ndt_i2r_*       /localization/pose_estimator/initial_to_result_relative_pose
                  how far the solution moved from the initial guess, DECOMPOSED PER AXIS -- the only
                  internal quantity that matches this study's per-axis target structure.
  ndt_cov_*       /localization/pose_estimator/pose_with_covariance
                  the covariance the estimator actually publishes. With the default
                  covariance_estimation_type: 0 this is a hand-set constant; logging it turns that
                  from a config inference into a measurement, and makes a future MULTI_NDT_SCORE
                  comparison possible.

These three are UNVERIFIED as model inputs: no evidence yet that they add information beyond the
existing 10 signals. They are recorded so the question can be asked later; the main results must
stand without them.
"""

import argparse
import csv
import json
import math
import statistics
import time

from autoware_internal_debug_msgs.msg import Float32Stamped
from autoware_internal_debug_msgs.msg import Int32Stamped
from autoware_vehicle_msgs.msg import VelocityReport
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


def stamp_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def angle_diff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def stat(values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return (math.nan, math.nan, math.nan, math.nan)
    return (
        statistics.fmean(values),
        statistics.pstdev(values) if len(values) > 1 else 0.0,
        min(values),
        max(values),
    )


class LaneRouteBaseline(Node):
    def __init__(self, args):
        super().__init__("carla_lane_route_baseline")
        self.args = args
        self.gt = None
        self.ndt = None
        self.velocity = None
        self.tp = math.nan
        self.iteration = math.nan
        self.nvtl = math.nan
        self.i2r = None
        self.ndt_cov = None
        self.route_state = {}
        self.ndt_times = []
        self.ndt_gap_latest = 0.0
        self.prev_ndt_xy = None
        self.ndt_jump_latest = 0.0
        self.rows = []
        self.trigger_result = "NOT_TRIGGERED"
        self.trigger_client = self.create_client(Trigger, "/carla_pilot/run_lane_route")

        qos = QoSProfile(
            depth=100,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(PoseWithCovarianceStamped, "/sensing/gnss/pose_with_covariance", self.on_gt, qos)
        self.create_subscription(PoseStamped, "/localization/pose_estimator/pose", self.on_ndt, qos)
        self.create_subscription(VelocityReport, "/vehicle/status/velocity_status", self.on_velocity, qos)
        self.create_subscription(Float32Stamped, "/localization/pose_estimator/transform_probability", self.on_tp, qos)
        self.create_subscription(Int32Stamped, "/localization/pose_estimator/iteration_num", self.on_iter, qos)
        # FIX 2026-08-01: these three were first added with the RELIABLE depth-100 profile above.
        # A RELIABLE reader applies back-pressure to the writer, and ndt_scan_matcher publishes from
        # its own optimisation callback -- so a slow Python logger stalled the NDT loop. Measured:
        # NDT output 117 -> 10 messages per 8 s probe, iterations pinned 7.8 -> 29.3 of the 30 max,
        # transform_probability 7.60 -> 2.57, and the pose drifted 0.22 -> 4.36 m while STATIONARY.
        # A passive logger must never back-pressure the estimator: BEST_EFFORT, depth 1.
        qos_obs = QoSProfile(depth=1,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(Float32Stamped,
                                 "/localization/pose_estimator/nearest_voxel_transformation_likelihood",
                                 self.on_nvtl, qos_obs)
        self.create_subscription(PoseStamped,
                                 "/localization/pose_estimator/initial_to_result_relative_pose",
                                 self.on_i2r, qos_obs)
        self.create_subscription(PoseWithCovarianceStamped,
                                 "/localization/pose_estimator/pose_with_covariance",
                                 self.on_ndt_cov, qos_obs)
        # v3 (2026-09-28): NDT execution time, BEST_EFFORT like the other observers
        self.exe_time = math.nan
        self.create_subscription(Float32Stamped,
                                 "/localization/pose_estimator/exe_time_ms",
                                 self.on_exe_time, qos_obs)
        self.create_subscription(String, "/carla_pilot/route_state", self.on_route_state, qos)
        self.create_timer(0.1, self.record)

    def on_gt(self, msg):
        self.gt = msg

    def on_ndt(self, msg):
        t = stamp_sec(msg.header.stamp)
        if self.ndt_times:
            self.ndt_gap_latest = t - self.ndt_times[-1]
        self.ndt_times.append(t)
        xy = (msg.pose.position.x, msg.pose.position.y)
        if self.prev_ndt_xy is not None:
            self.ndt_jump_latest = math.hypot(xy[0] - self.prev_ndt_xy[0], xy[1] - self.prev_ndt_xy[1])
        self.prev_ndt_xy = xy
        self.ndt = msg

    def on_velocity(self, msg):
        self.velocity = msg

    def on_tp(self, msg):
        self.tp = float(msg.data)

    def on_iter(self, msg):
        self.iteration = int(msg.data)

    def on_nvtl(self, msg):
        self.nvtl = float(msg.data)

    def on_i2r(self, msg):
        self.i2r = msg

    def on_ndt_cov(self, msg):
        self.ndt_cov = msg

    def on_exe_time(self, msg):
        self.exe_time = float(msg.data)

    def on_route_state(self, msg):
        try:
            self.route_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.route_state = {}

    def record(self):
        if self.gt is None or self.ndt is None or not self.route_state:
            return
        gt_pose = self.gt.pose.pose
        ndt_pose = self.ndt.pose
        gt_yaw = yaw_of(gt_pose.orientation)
        ndt_yaw = yaw_of(ndt_pose.orientation)
        dx = ndt_pose.position.x - gt_pose.position.x
        dy = ndt_pose.position.y - gt_pose.position.y
        c = math.cos(gt_yaw)
        s = math.sin(gt_yaw)
        lon = c * dx + s * dy
        lat = -s * dx + c * dy
        phase = self.route_state.get("phase", "before")
        if phase == "route":
            phase_out = "route"
        elif phase == "brake":
            phase_out = "brake"
        elif phase == "after":
            phase_out = "after"
        else:
            phase_out = "before"
        self.rows.append(
            {
                "timestamp": stamp_sec(self.gt.header.stamp),
                "phase": phase_out,
                "route_progress_m": self.route_state.get("route_progress", math.nan),
                "route_waypoint_index": self.route_state.get("waypoint_index", -1),
                "carla_ego_x": self.route_state.get("ego_x", math.nan),
                "carla_ego_y": self.route_state.get("ego_y", math.nan),
                "carla_ego_z": self.route_state.get("ego_z", math.nan),
                "carla_ego_yaw": self.route_state.get("ego_yaw", math.nan),
                "carla_ego_speed": self.route_state.get("ego_speed", math.nan),
                "lane_centerline_lateral_deviation": self.route_state.get("lane_lateral_deviation", math.nan),
                "route_heading_error": self.route_state.get("route_heading_error", math.nan),
                "route_v_ref_mps": self.route_state.get("v_ref_mps", math.nan),
                "route_speed_error_mps": self.route_state.get("speed_error_mps", math.nan),
                "route_speed_profile_enabled": bool(self.route_state.get("speed_profile_enabled", False)),
                "throttle": self.route_state.get("throttle", math.nan),
                "brake": self.route_state.get("brake", math.nan),
                "steering": self.route_state.get("steer", math.nan),
                "gt_x": gt_pose.position.x,
                "gt_y": gt_pose.position.y,
                "gt_yaw": math.degrees(gt_yaw),
                "ndt_x": ndt_pose.position.x,
                "ndt_y": ndt_pose.position.y,
                "ndt_yaw": math.degrees(ndt_yaw),
                "raw_longitudinal_residual": lon,
                "raw_lateral_residual": lat,
                "raw_yaw_residual": math.degrees(angle_diff(ndt_yaw, gt_yaw)),
                "raw_2d_residual": math.hypot(dx, dy),
                "transform_probability": self.tp,
                "iteration_num": self.iteration,
                "ndt_timestamp": stamp_sec(self.ndt.header.stamp),
                "ndt_output_gap": self.ndt_gap_latest,
                "ndt_inter_sample_jump": self.ndt_jump_latest,
                "collision": bool(self.route_state.get("collision", False)),
                "safety_abort": bool(self.route_state.get("safety_abort", False)),
                "abort_reason": self.route_state.get("abort_reason", ""),
                # ---- v2 additions (appended; v1 column order above is untouched) ----
                "ndt_nvtl": self.nvtl,
                "ndt_i2r_dx": (self.i2r.pose.position.x if self.i2r else math.nan),
                "ndt_i2r_dy": (self.i2r.pose.position.y if self.i2r else math.nan),
                "ndt_i2r_dyaw": (math.degrees(yaw_of(self.i2r.pose.orientation))
                                 if self.i2r else math.nan),
                "ndt_cov_xx": (self.ndt_cov.pose.covariance[0] if self.ndt_cov else math.nan),
                "ndt_cov_yy": (self.ndt_cov.pose.covariance[7] if self.ndt_cov else math.nan),
                "ndt_cov_yawyaw": (self.ndt_cov.pose.covariance[35] if self.ndt_cov else math.nan),
                # ---- v3 additions (appended; the v2 column order above is untouched) ----
                "ndt_cov_xy": (self.ndt_cov.pose.covariance[1] if self.ndt_cov else math.nan),
                "ndt_cov_stamp": (stamp_sec(self.ndt_cov.header.stamp) if self.ndt_cov else math.nan),
                "ndt_exe_time_ms": self.exe_time,
            }
        )

    def wait_collect(self, duration):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.2)

    def go_check(self):
        rows = self.rows[-max(1, int(self.args.precheck_sec * 8)) :]
        if len(rows) < self.args.min_go_samples:
            return False, f"too_few_samples={len(rows)}"
        tp_mean = stat([r["transform_probability"] for r in rows])[0]
        d2_max = max(r["raw_2d_residual"] for r in rows)
        yaw_abs = max(abs(r["raw_yaw_residual"]) for r in rows)
        gaps = [r["ndt_output_gap"] for r in rows if r["ndt_output_gap"] > self.args.gap_threshold]
        if tp_mean < self.args.min_tp:
            return False, f"tp_mean={tp_mean:.3f}"
        if d2_max > self.args.max_d2:
            return False, f"d2_max={d2_max:.3f}"
        if yaw_abs > self.args.max_yaw_deg:
            return False, f"yaw_abs={yaw_abs:.3f}"
        if gaps:
            return False, f"gaps={len(gaps)}"
        return True, f"tp_mean={tp_mean:.3f} d2_max={d2_max:.3f} yaw_abs={yaw_abs:.3f} gaps=0"

    def trigger(self):
        if not self.trigger_client.wait_for_service(timeout_sec=10.0):
            self.trigger_result = "SERVICE_UNAVAILABLE"
            return False
        future = self.trigger_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        if result is None:
            self.trigger_result = "NO_RESPONSE"
            return False
        self.trigger_result = f"success={result.success} message={result.message!r}"
        return bool(result.success)

    def wait_done(self):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < self.args.max_run_sec:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.route_state.get("done") and self.route_state.get("phase") == "after":
                self.wait_collect(self.args.after_done_sec)
                return True
        return False

    def write_csv(self):
        if not self.rows:
            return
        with open(self.args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)

    def summarize_rows(self, label, rows):
        return (
            f"{label}: n={len(rows)} "
            f"lon_mean={stat([r['raw_longitudinal_residual'] for r in rows])[0]:.3f} "
            f"lon_std={stat([r['raw_longitudinal_residual'] for r in rows])[1]:.3f} "
            f"lat_mean={stat([r['raw_lateral_residual'] for r in rows])[0]:.3f} "
            f"lat_std={stat([r['raw_lateral_residual'] for r in rows])[1]:.3f} "
            f"yaw_abs_max={stat([abs(r['raw_yaw_residual']) for r in rows])[3]:.3f} "
            f"tp_mean={stat([r['transform_probability'] for r in rows])[0]:.3f} "
            f"tp_min={stat([r['transform_probability'] for r in rows])[2]:.3f} "
            f"iter_mean={stat([r['iteration_num'] for r in rows])[0]:.2f} "
            f"iter_max={stat([r['iteration_num'] for r in rows])[3]:.0f}"
        )

    def bin_summary(self):
        route_rows = [r for r in self.rows if r["phase"] == "route"]
        if not route_rows:
            return []
        max_s = max(r["route_progress_m"] for r in route_rows)
        lines = []
        b = 0.0
        while b <= max_s + 1e-6:
            rows = [r for r in route_rows if b <= r["route_progress_m"] < b + 10.0]
            if rows:
                lines.append(
                    f"BIN {b:.0f}-{b+10:.0f}m n={len(rows)} "
                    f"lon_mean={stat([r['raw_longitudinal_residual'] for r in rows])[0]:.3f} "
                    f"lon_std={stat([r['raw_longitudinal_residual'] for r in rows])[1]:.3f} "
                    f"lat_mean={stat([r['raw_lateral_residual'] for r in rows])[0]:.3f} "
                    f"lat_std={stat([r['raw_lateral_residual'] for r in rows])[1]:.3f} "
                    f"yaw_mean={stat([r['raw_yaw_residual'] for r in rows])[0]:.3f} "
                    f"yaw_std={stat([r['raw_yaw_residual'] for r in rows])[1]:.3f} "
                    f"tp_mean={stat([r['transform_probability'] for r in rows])[0]:.3f} "
                    f"tp_min={stat([r['transform_probability'] for r in rows])[2]:.3f} "
                    f"iter_mean={stat([r['iteration_num'] for r in rows])[0]:.2f} "
                    f"lane_dev_mean={stat([abs(r['lane_centerline_lateral_deviation']) for r in rows])[0]:.3f} "
                    f"lane_dev_max={stat([abs(r['lane_centerline_lateral_deviation']) for r in rows])[3]:.3f}"
                )
            b += 10.0
        return lines

    def write_summary(self, go_ok, go_reason, done):
        route_rows = [r for r in self.rows if r["phase"] == "route"]
        before = [r for r in self.rows if r["phase"] == "before"]
        after = [r for r in self.rows if r["phase"] == "after"][-max(1, int(self.args.post_summary_sec * 8)) :]
        max_progress = max((r["route_progress_m"] for r in self.rows), default=0.0)
        route_length = max((self.route_state.get("route_length", 0.0), max_progress))
        max_speed = max((r["carla_ego_speed"] for r in self.rows), default=0.0)
        max_lane_dev = max((abs(r["lane_centerline_lateral_deviation"]) for r in self.rows if math.isfinite(r["lane_centerline_lateral_deviation"])), default=math.nan)
        gaps = [r["ndt_output_gap"] for r in self.rows if r["ndt_output_gap"] > self.args.gap_threshold]
        max_jump = max((r["ndt_inter_sample_jump"] for r in self.rows), default=0.0)
        aborts = [r["abort_reason"] for r in self.rows if r["abort_reason"]]
        route_completion = done and not aborts and max_progress >= route_length - 2.0
        if self.args.startup_verify_only:
            verdict = "STARTUP VERIFIED" if go_ok else "STARTUP VERIFY FAIL"
        else:
            gross_fail = bool(gaps) or max_jump > 10.0 or aborts or not route_completion
            verdict = "ROUTE CONTROL FAIL" if aborts or not route_completion else "ROUTE BASELINE PASS"
            if gross_fail and verdict != "ROUTE CONTROL FAIL":
                verdict = "ROUTE BASELINE FAIL"
        lines = [
            f"RUN {self.args.run_id}",
            f"GO ok={go_ok} reason={go_reason}",
            f"TRIGGER {self.trigger_result}",
            f"ROUTE completion={route_completion} max_progress={max_progress:.3f} route_length={route_length:.3f} max_speed={max_speed:.3f} max_lane_dev={max_lane_dev:.3f}",
            f"SPEED_PROFILE v_ref_mean={stat([r['route_v_ref_mps'] for r in route_rows])[0]:.3f} v_ref_max={stat([r['route_v_ref_mps'] for r in route_rows])[3]:.3f} actual_mean={stat([r['carla_ego_speed'] for r in route_rows])[0]:.3f}",
            f"NDT gaps={len(gaps)} max_gap={max(gaps) if gaps else 0.0:.3f} max_jump={max_jump:.3f}",
            f"ABORT reasons={sorted(set(aborts))}",
            self.summarize_rows("BEFORE", before),
            self.summarize_rows("ROUTE", route_rows),
            self.summarize_rows("AFTER_LAST", after),
            *self.bin_summary(),
            f"VERDICT {verdict}",
        ]
        with open(self.args.summary, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--precheck-sec", type=float, default=8.0)
    parser.add_argument("--post-summary-sec", type=float, default=10.0)
    parser.add_argument("--after-done-sec", type=float, default=2.0)
    parser.add_argument("--max-run-sec", type=float, default=120.0)
    parser.add_argument("--min-go-samples", type=int, default=50)
    parser.add_argument("--min-tp", type=float, default=6.5)
    parser.add_argument("--max-d2", type=float, default=2.0)
    parser.add_argument("--max-yaw-deg", type=float, default=5.0)
    parser.add_argument("--gap-threshold", type=float, default=0.5)
    parser.add_argument(
        "--startup-verify-only",
        action="store_true",
        help="Stop after pre-route GO gate; do not trigger or drive the route.",
    )
    args = parser.parse_args()
    rclpy.init()
    node = LaneRouteBaseline(args)
    node.wait_collect(args.precheck_sec)
    go_ok, go_reason = node.go_check()
    done = False
    if go_ok and args.startup_verify_only:
        node.trigger_result = "SKIPPED startup_verify_only"
    elif go_ok and node.trigger():
        done = node.wait_done()
    elif go_ok:
        go_ok = False
        go_reason = f"trigger_failed:{node.trigger_result}"
    node.write_csv()
    node.write_summary(go_ok, go_reason, done)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
