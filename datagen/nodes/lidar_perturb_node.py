#!/usr/bin/env python3
"""CARLA LiDAR perturbation node — route-progress scheduled.

Ports the AWSIM perturbation pipeline (datagen/scenario_runner/
lidar_dropout_node.py) to CARLA with ONLY the application path changed:

  * input/output topics are the CARLA bridge LiDAR topics
  * the schedule axis is ROUTE PROGRESS [m] (from /carla_pilot/route_state),
    not wall-clock relative to a control trigger
  * activation is recorded against route progress

The perturbation MATH is reused unchanged by importing the AWSIM kernels
(keep_indices_for_*, apply_yaw_rotation, apply_lateral_translation,
parse_schedule, schedule_value_at, sector_center_deg). No new modes, no new
severities, no new magnitudes are introduced here. Seed default matches AWSIM.

Schedule format: 's_m:value' step-wise constant, where s_m is route progress in
metres. These progress values come from the manifest column
`carla_schedule_progress_m`, produced by mapping each AWSIM `t_sec` onset through
the route's v_ref-integral time->progress curve (mapping_method=route_vref_integral
in carla_scenario_manifest_v1.csv) -- NOT a fixed 2.0 m/s. dropout value=drop
ratio [0,1]; range_clip value=max radius [m]; sector_remove value=removed sector
width [deg]; sector_thin value=removal ratio [0,1]; yaw_rotate value=angle [deg];
translate_y value=lateral offset [m]. (Same per-mode validation ranges as AWSIM.)
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy,
)
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

# --- reuse the exact AWSIM perturbation kernels -----------------------------
_SR = str(Path(__file__).resolve().parent / "awsim_kernels")
if _SR not in sys.path:
    sys.path.insert(0, _SR)
import lidar_dropout_node as awsim  # noqa: E402

# --- v2 addition: the NON-RIGID category (see nonrigid_kernels.py) ----
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nonrigid_kernels as nrk  # noqa: E402

NONRIGID_MODES = ("yaw_shear", "lateral_shear")
LON_MODES = ("translate_x", "longitudinal_shear")          # longitudinal modes
ALL_MODES = ["dropout", "range_clip", "sector_remove", "sector_thin",
             "yaw_rotate", "translate_y"] + list(NONRIGID_MODES) + list(LON_MODES)

ZERO_INACTIVE = ("dropout", "sector_remove", "sector_thin", "yaw_rotate", "translate_y",
                 "yaw_shear", "lateral_shear", "translate_x", "longitudinal_shear")


class CarlaLidarPerturbNode(Node):
    def __init__(self, args):
        super().__init__("carla_lidar_perturb_node")
        self._a = args
        self._schedule = awsim.parse_schedule(args.schedule, args.mode)  # reuse + validate
        if args.mode in NONRIGID_MODES:                    # v2: the AWSIM parser knows nothing
            for _, _v in self._schedule:                   # about these, so validate here
                nrk.validate(args.mode, _v)
        if args.mode in LON_MODES:                          # validated by nrk as well
            for _, _v in self._schedule:
                nrk.validate(args.mode, _v)
        self._rng = np.random.default_rng(args.seed)
        self._rows = []
        self._log = Path(args.log_csv) if args.log_csv else None
        if self._log:
            self._log.parent.mkdir(parents=True, exist_ok=True)

        # route-progress state from the CARLA bridge
        self._progress = None
        self._phase = "before"

        qos_be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                            durability=QoSDurabilityPolicy.VOLATILE)
        qos_rel = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self._pub = self.create_publisher(PointCloud2, args.output_topic, qos_be)
        self.create_subscription(PointCloud2, args.input_topic, self._cb_cloud, qos_be)
        self.create_subscription(String, args.route_state_topic, self._cb_route, qos_rel)
        self.get_logger().info(
            f"CarlaLidarPerturbNode ready. mode={args.mode} "
            f"in={args.input_topic} out={args.output_topic}\n"
            f"progress_schedule={self._schedule} seed={args.seed} "
            f"route_state={args.route_state_topic} activate_phase={args.activate_phase}"
        )

    def _cb_route(self, msg):
        try:
            st = json.loads(msg.data)
            self._progress = float(st.get("route_progress", math.nan))
            self._phase = st.get("phase", self._phase)
        except Exception:
            pass

    def _armed(self):
        # active only once the route is actually being driven (progress meaningful)
        if self._progress is None or not math.isfinite(self._progress):
            return False
        if self._a.activate_phase and self._phase not in ("route", "brake", "after"):
            return False
        return True

    def _cb_cloud(self, msg):
        default = 0.0 if self._a.mode in ZERO_INACTIVE else math.inf
        if not self._armed():
            s_rel = -1.0
            target = default
        else:
            s_rel = self._progress
            target = awsim.schedule_value_at(self._schedule, s_rel, default)

        n_in = int(msg.width) * int(msg.height)
        step = int(msg.point_step)
        raw = bytes(msg.data)
        raw_points = (np.empty((0, step), dtype=np.uint8) if n_in == 0
                      else np.frombuffer(raw, dtype=np.uint8).reshape(-1, step))

        keep = None
        new_data = None
        tval = {"target_ratio": math.nan, "target_radius_m": math.nan,
                "sector_width_deg": math.nan, "yaw_rotation_deg": math.nan,
                "lateral_offset_m": math.nan,
                "yaw_shear_deg_at_ref": math.nan, "lateral_shear_m_at_ref": math.nan,
                "longitudinal_offset_m": math.nan, "longitudinal_shear_m_at_ref": math.nan}
        t0 = time.perf_counter()
        m = self._a.mode
        if m == "dropout":
            keep = awsim.keep_indices_for_dropout(n_in, target, self._rng)
            tval["target_ratio"] = target
        elif m == "range_clip":
            keep = awsim.keep_indices_for_range_clip(msg, raw_points, target)
            tval["target_radius_m"] = target
        elif m == "sector_remove":
            keep = awsim.keep_indices_for_sector_remove(msg, raw_points, self._a.sector_center_deg, target)
            tval["sector_width_deg"] = target
        elif m == "sector_thin":
            keep = awsim.keep_indices_for_sector_thin(msg, raw_points, self._a.sector_center_deg,
                                                      self._a.sector_thin_width, target, self._rng)
            tval["target_ratio"] = target
            tval["sector_width_deg"] = self._a.sector_thin_width
        elif m == "yaw_rotate":
            new_data = awsim.apply_yaw_rotation(msg, raw_points, target)
            n_out = n_in
            tval["yaw_rotation_deg"] = target
        elif m == "translate_y":
            new_data = awsim.apply_lateral_translation(msg, raw_points, target)
            n_out = n_in
            tval["lateral_offset_m"] = target
        elif m == "yaw_shear":                                          # v2: NON-RIGID
            new_data = nrk.apply_yaw_shear(msg, raw_points, target, self._a.shear_ref_range_m)
            n_out = n_in
            tval["yaw_shear_deg_at_ref"] = target
        elif m == "lateral_shear":                                      # v2: NON-RIGID
            new_data = nrk.apply_lateral_shear(msg, raw_points, target, self._a.shear_ref_range_m)
            n_out = n_in
            tval["lateral_shear_m_at_ref"] = target
        elif m == "translate_x":                                        # rigid lon control
            new_data = nrk.apply_longitudinal_translation(msg, raw_points, target)
            n_out = n_in
            tval["longitudinal_offset_m"] = target
        elif m == "longitudinal_shear":                                 # NON-RIGID lon
            new_data = nrk.apply_longitudinal_shear(msg, raw_points, target, self._a.shear_ref_range_m)
            n_out = n_in
            tval["longitudinal_shear_m_at_ref"] = target
        else:
            raise RuntimeError(f"unsupported mode: {m}")

        if keep is not None:
            n_out = int(keep.size)
            new_data = msg.data if n_out == n_in else raw_points[keep].tobytes()

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        keep_ratio = (n_out / n_in) if n_in else 0.0
        removed_ratio = 1.0 - keep_ratio if n_in else 0.0

        out = PointCloud2()
        out.header = msg.header           # preserve stamp + frame_id exactly
        out.height = 1
        out.width = n_out
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = msg.point_step * n_out
        out.is_dense = msg.is_dense
        out.data = bytes(new_data)
        self._pub.publish(out)

        self._rows.append({
            "route_progress_m": round(s_rel, 4),
            "phase": self._phase,
            "armed": int(self._armed()),
            "mode": m,
            "target_value": round(target, 4) if math.isfinite(target) else "inf",
            "target_ratio": round(tval["target_ratio"], 4) if math.isfinite(tval["target_ratio"]) else "",
            "target_radius_m": round(tval["target_radius_m"], 4) if math.isfinite(tval["target_radius_m"]) else "",
            "sector_center_deg": self._a.sector_center_deg if m in ("sector_remove", "sector_thin") else "",
            "sector_width_deg": round(tval["sector_width_deg"], 4) if math.isfinite(tval["sector_width_deg"]) else "",
            "yaw_rotation_deg": round(tval["yaw_rotation_deg"], 4) if math.isfinite(tval["yaw_rotation_deg"]) else "",
            "lateral_offset_m": round(tval["lateral_offset_m"], 4) if math.isfinite(tval["lateral_offset_m"]) else "",
            "yaw_shear_deg_at_ref": (round(tval["yaw_shear_deg_at_ref"], 4)
                                     if math.isfinite(tval["yaw_shear_deg_at_ref"]) else ""),
            "lateral_shear_m_at_ref": (round(tval["lateral_shear_m_at_ref"], 4)
                                       if math.isfinite(tval["lateral_shear_m_at_ref"]) else ""),
            "longitudinal_offset_m": (round(tval["longitudinal_offset_m"], 4)
                                      if math.isfinite(tval["longitudinal_offset_m"]) else ""),
            "longitudinal_shear_m_at_ref": (round(tval["longitudinal_shear_m_at_ref"], 4)
                                            if math.isfinite(tval["longitudinal_shear_m_at_ref"]) else ""),
            "shear_ref_range_m": (self._a.shear_ref_range_m
                                  if m in NONRIGID_MODES + LON_MODES else ""),
            "n_in": n_in,
            "n_out": n_out,
            "actual_keep_ratio": round(keep_ratio, 6),
            "actual_removed_ratio": round(removed_ratio, 6),
            "elapsed_ms": round(elapsed_ms, 4),
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nsec": int(msg.header.stamp.nanosec),
        })
        if self._a.verbose and self._armed():
            b = int(s_rel // 10)
            if b != getattr(self, "_lb", -1):
                self._lb = b
                self.get_logger().info(
                    f"progress={s_rel:7.2f}m phase={self._phase} mode={m} "
                    f"target={target} n_in={n_in} n_out={n_out} removed={removed_ratio*100:.2f}%")

    def flush(self):
        if not self._log or not self._rows:
            return
        with self._log.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            w.writeheader()
            w.writerows(self._rows)
        self.get_logger().info(f"wrote {len(self._rows)} scan rows -> {self._log}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=ALL_MODES, required=True)
    p.add_argument("--shear-ref-range-m", type=float, default=nrk.DEFAULT_REF_RANGE_M,
                   help="reference range at which the yaw_shear/lateral_shear severity is quoted")
    p.add_argument("--input-topic", default="/sensing/lidar/top/pointcloud_before_sync")
    p.add_argument("--output-topic", default="/sensing/lidar/top/pointcloud_before_sync_dropout")
    p.add_argument("--route-state-topic", default="/carla_pilot/route_state")
    p.add_argument("--schedule", required=True,
                   help="route-progress schedule 's_m:value' (step-wise constant); "
                        "use the manifest carla_schedule_progress_m (route_vref_integral)")
    p.add_argument("--sector-direction", choices=["front", "left", "right", "rear",
                                                  "front_left", "front_right"], default="left")
    p.add_argument("--sector-center-deg", type=float, default=None,
                   help="explicit sector center [deg]; overrides --sector-direction")
    p.add_argument("--sector-thin-width", type=float, default=160.0)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--activate-phase", action="store_true", default=True,
                   help="only perturb once route_state phase is route/brake/after")
    p.add_argument("--log-csv", type=str, default=None)
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args()

    # sector centers (extends AWSIM front/left/right with rear/front_left/front_right
    # using the AWSIM node's documented --sector-center-deg override mechanism)
    centers = {"front": 0.0, "left": 90.0, "right": -90.0,
               "rear": 180.0, "front_left": 45.0, "front_right": -45.0}
    if args.sector_center_deg is None:
        args.sector_center_deg = centers[args.sector_direction]

    rclpy.init()
    node = CarlaLidarPerturbNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.flush()
        except Exception as exc:
            print(f"flush failed: {exc}", file=sys.stderr)
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
