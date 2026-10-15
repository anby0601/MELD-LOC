#!/usr/bin/env python3
"""Short CARLA NDT health probe for recovery runs.

Prints the legacy summary lines consumed by recovery tooling, plus a
PROBE_RESULT status that distinguishes execution success with no pose from
unhealthy NDT output.
"""

import argparse
import math
import statistics
import sys
import time

import rclpy
from autoware_internal_debug_msgs.msg import Float32Stamped
from autoware_internal_debug_msgs.msg import Int32Stamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy


def yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def adiff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def stat(values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return (math.nan, math.nan, math.nan)
    return (statistics.fmean(values), min(values), max(values))


class Probe(Node):
    def __init__(self):
        super().__init__("carla_health_probe")
        self.gt = None
        self.ndt = None
        self.tp = math.nan
        self.iteration = math.nan
        self.ndt_times = []
        self.rows = []
        qos = QoSProfile(
            depth=100,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/sensing/gnss/pose_with_covariance",
            self._on_gt,
            qos,
        )
        self.create_subscription(PoseStamped, "/localization/pose_estimator/pose", self._on_ndt, qos)
        self.create_subscription(
            Float32Stamped,
            "/localization/pose_estimator/transform_probability",
            self._on_tp,
            qos,
        )
        self.create_subscription(Int32Stamped, "/localization/pose_estimator/iteration_num", self._on_iter, qos)
        self.create_timer(0.1, self._tick)

    def _on_gt(self, msg):
        self.gt = msg

    def _on_ndt(self, msg):
        self.ndt_times.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.ndt = msg

    def _on_tp(self, msg):
        self.tp = float(msg.data)

    def _on_iter(self, msg):
        self.iteration = int(msg.data)

    def _tick(self):
        if self.gt is None or self.ndt is None:
            return
        gt = self.gt.pose.pose
        ndt = self.ndt.pose
        gt_yaw = yaw(gt.orientation)
        dx = ndt.position.x - gt.position.x
        dy = ndt.position.y - gt.position.y
        c = math.cos(gt_yaw)
        s = math.sin(gt_yaw)
        self.rows.append(
            (
                self.tp,
                math.hypot(dx, dy),
                c * dx + s * dy,
                -s * dx + c * dy,
                math.degrees(adiff(yaw(ndt.orientation), gt_yaw)),
                self.iteration,
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument("--min-tp", type=float, default=6.5)
    parser.add_argument("--max-d2-mean", type=float, default=2.0)
    parser.add_argument("--max-yaw-deg", type=float, default=5.0)
    parser.add_argument("--gap-threshold", type=float, default=0.5)
    args = parser.parse_args()

    rclpy.init()
    node = Probe()
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < args.duration_sec:
        rclpy.spin_once(node, timeout_sec=0.2)

    rows = node.rows
    gaps = sum(
        1 for i in range(1, len(node.ndt_times)) if node.ndt_times[i] - node.ndt_times[i - 1] > args.gap_threshold
    )
    max_gap = max(
        [node.ndt_times[i] - node.ndt_times[i - 1] for i in range(1, len(node.ndt_times))]
        or [0.0]
    )
    tp = stat([r[0] for r in rows])
    d2 = stat([r[1] for r in rows])
    lon = stat([r[2] for r in rows])
    lat = stat([r[3] for r in rows])
    yaw_abs = [abs(r[4]) for r in rows if math.isfinite(r[4])]
    it = stat([r[5] for r in rows])

    print(f"samples={len(rows)} ndt_msgs={len(node.ndt_times)}")
    print(f"TP mean/min={tp[0]:.3f}/{tp[1]:.3f}  d2 mean/min/max={d2[0]:.3f}/{d2[1]:.3f}/{d2[2]:.3f}")
    print(f"lon={lon[0]:.3f} lat={lat[0]:.3f}")
    print(f"|yaw|max={max(yaw_abs) if yaw_abs else math.nan:.3f}  iter mean/max={it[0]:.2f}/{it[2]:.0f} gaps>0.5s={gaps} max_gap={max_gap:.3f}")

    if not rows:
        status = "NDT_NO_POSE"
        reason = "no_paired_gt_ndt_samples"
    elif len(rows) < args.min_samples:
        status = "NDT_NO_POSE"
        reason = f"too_few_samples={len(rows)}"
    elif tp[0] < args.min_tp:
        status = "NDT_UNHEALTHY"
        reason = f"tp_mean={tp[0]:.3f}"
    elif d2[0] > args.max_d2_mean:
        status = "NDT_UNHEALTHY"
        reason = f"d2_mean={d2[0]:.3f}"
    elif yaw_abs and max(yaw_abs) > args.max_yaw_deg:
        status = "NDT_UNHEALTHY"
        reason = f"yaw_abs={max(yaw_abs):.3f}"
    elif gaps:
        status = "NDT_UNHEALTHY"
        reason = f"gaps={gaps}"
    else:
        status = "OK"
        reason = f"tp_mean={tp[0]:.3f}"

    print(f"PROBE_RESULT status={status} reason={reason}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
