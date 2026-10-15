

#!/usr/bin/env python3
"""Initialize Autoware localization once from the current CARLA GNSS/GT pose.

This uses the official /localization/initialize service with DIRECT method.
It is intended for logging_simulation mode, where pose_initializer stop-check is
disabled by official launch logic. The script does not publish during the run
after the one initialization service call.
"""

import argparse
import math
import time

import rclpy
from autoware_localization_msgs.srv import InitializeLocalization
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


class DirectInit(Node):
    def __init__(self):
        super().__init__("carla_direct_init")
        self.gt_msg = None
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/sensing/gnss/pose_with_covariance",
            self._on_gt,
            qos,
        )
        self.client = self.create_client(InitializeLocalization, "/localization/initialize")

    def _on_gt(self, msg: PoseWithCovarianceStamped):
        self.gt_msg = msg


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--cov-xy", type=float, default=1.0)
    parser.add_argument("--cov-yaw", type=float, default=0.25)
    args = parser.parse_args()

    rclpy.init()
    node = DirectInit()
    t0 = time.time()
    while node.gt_msg is None and time.time() - t0 < args.timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.2)
    if node.gt_msg is None:
        print("INIT_RESULT failure reason=NO_GT")
        return
    if not node.client.wait_for_service(timeout_sec=args.timeout_sec):
        print("INIT_RESULT failure reason=SERVICE_UNAVAILABLE")
        return

    pose = PoseWithCovarianceStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = node.get_clock().now().to_msg()
    pose.pose.pose = node.gt_msg.pose.pose
    if args.yaw_offset_deg:
        yaw = yaw_from_quat(pose.pose.pose.orientation) + math.radians(args.yaw_offset_deg)
        pose.pose.pose.orientation.x = 0.0
        pose.pose.pose.orientation.y = 0.0
        pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.pose.orientation.w = math.cos(yaw / 2.0)

    cov = [0.0] * 36
    cov[0] = args.cov_xy
    cov[7] = args.cov_xy
    cov[35] = args.cov_yaw
    pose.pose.covariance = cov

    req = InitializeLocalization.Request()
    req.method = InitializeLocalization.Request.DIRECT
    req.pose_with_covariance = [pose]

    gt_stamp = node.gt_msg.header.stamp.sec + node.gt_msg.header.stamp.nanosec * 1e-9
    init_stamp = pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9
    yaw_deg = math.degrees(yaw_from_quat(pose.pose.pose.orientation))
    print(
        "INIT_REQUEST method=DIRECT "
        f"gt_stamp={gt_stamp:.9f} init_stamp={init_stamp:.9f} "
        f"x={pose.pose.pose.position.x:.3f} y={pose.pose.pose.position.y:.3f} "
        f"yaw_deg={yaw_deg:.3f} cov_xy={args.cov_xy:.3f} cov_yaw={args.cov_yaw:.3f}"
    )

    future = node.client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=60.0)
    result = future.result()
    if result and result.status.success:
        print(f"INIT_RESULT success code={result.status.code} message={result.status.message!r}")
    elif result:
        print(f"INIT_RESULT failure code={result.status.code} message={result.status.message!r}")
    else:
        print("INIT_RESULT failure reason=NO_RESPONSE")


if __name__ == "__main__":
    main()
