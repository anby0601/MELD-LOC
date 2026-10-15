

#!/usr/bin/env python3
"""Minimal CARLA -> Autoware bridge for Town01 NDT localization probes.

This intentionally publishes only the surfaces needed by localization:
clock, LiDAR PointCloud2, GNSS-like ego pose, and vehicle velocity.
"""

import argparse
import json
import math
import os
import queue
import time

import carla
from autoware_vehicle_msgs.msg import VelocityReport
import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tier4_vehicle_msgs.msg import ActuationCommandStamped


def ros_stamp(seconds_float):
    sec = int(seconds_float)
    return Time(sec=sec, nanosec=int((seconds_float - sec) * 1e9))


def yaw_to_quaternion(yaw):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def carla_transform_to_ros_pose(transform):
    pose = PoseWithCovarianceStamped()
    pose.header.frame_id = "map"
    pose.pose.pose.position.x = transform.location.x
    pose.pose.pose.position.y = -transform.location.y
    pose.pose.pose.position.z = transform.location.z
    yaw = math.radians(-transform.rotation.yaw)
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    pose.pose.pose.orientation.x = qx
    pose.pose.pose.orientation.y = qy
    pose.pose.pose.orientation.z = qz
    pose.pose.pose.orientation.w = qw
    pose.pose.covariance[0] = 0.01
    pose.pose.covariance[7] = 0.01
    pose.pose.covariance[14] = 0.01
    pose.pose.covariance[35] = 1.0
    return pose


def look_at_rotation(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    dz = dst.z - src.z
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def lidar_to_pointcloud2(measurement, stamp):
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.UINT8, count=1),
        PointField(name="return_type", offset=13, datatype=PointField.UINT8, count=1),
        PointField(name="channel", offset=14, datatype=PointField.UINT16, count=1),
    ]
    raw = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    count = raw.shape[0]
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("intensity", "u1"),
        ("return_type", "u1"),
        ("channel", "u2"),
    ]
    data = np.zeros(count, dtype=dtype)
    data["x"] = raw[:, 0]
    data["y"] = -raw[:, 1]
    data["z"] = raw[:, 2]
    data["intensity"] = (np.clip(raw[:, 3], 0.0, 1.0) * 255.0).astype(np.uint8)
    data["return_type"] = 0

    offset = 0
    channels = []
    for ch in range(64):
        n = measurement.get_point_count(ch)
        if n:
            channels.append(np.full(n, ch, dtype=np.uint16))
        offset += n
    if channels and offset == count:
        data["channel"] = np.concatenate(channels)

    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id="velodyne_top")
    msg.height = 1
    msg.width = count
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * count
    msg.data = data.tobytes()
    msg.is_dense = False
    return msg


class MinimalCarlaBridge:
    def __init__(self, args):
        self.args = args
        rclpy.init()
        self.node = rclpy.create_node("carla_minimal_lidar_bridge")
        lidar_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.clock_pub = self.node.create_publisher(Clock, "/clock", 10)
        self.lidar_pub = self.node.create_publisher(
            PointCloud2, "/sensing/lidar/top/pointcloud_before_sync", lidar_qos
        )
        self.pose_pub = self.node.create_publisher(
            PoseWithCovarianceStamped, "/sensing/gnss/pose_with_covariance", 10
        )
        self.vel_pub = self.node.create_publisher(
            VelocityReport, "/vehicle/status/velocity_status", 10
        )
        self.control_pub = self.node.create_publisher(
            ActuationCommandStamped, "/carla_pilot/control_cmd", 10
        )
        self.motion_srv = self.node.create_service(
            Trigger, "/carla_pilot/run_short_motion", self._run_short_motion
        )
        self.route_srv = self.node.create_service(
            Trigger, "/carla_pilot/run_lane_route", self._run_lane_route
        )
        self.route_state_pub = self.node.create_publisher(String, "/carla_pilot/route_state", 10)
        self.lidar_queue = queue.Queue(maxsize=2)
        self.actors = []
        self.collision_detected = False
        self.motion_active = False
        self.motion_done = False
        self.motion_start_time = None
        self.drive_start_time = None
        self.brake_start_time = None
        self.initial_location = None
        self.max_speed = 0.0
        self.max_distance = 0.0
        self.current_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        self.spectator = None
        self.route = self._load_route(args.route_file)
        self.route_active = False
        self.route_done = False
        self.route_phase = "before"
        self.route_trigger_time = None
        self.route_brake_start_time = None
        self.route_abort_reason = ""
        self.route_prev_progress = 0.0
        self.route_last_progress_time = None
        self.route_lateral_exceed_since = None
        self.route_speed_integral = 0.0
        self.route_prev_speed_error = 0.0
        self.route_prev_control_time = None
        self.route_v_ref = 0.0
        self.route_speed_error = 0.0
        self.route_state = {}

    def _load_route(self, path):
        if not path:
            return []
        with open(path) as f:
            data = json.load(f)
        return data["waypoints"] if isinstance(data, dict) else data

    def setup(self):
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(self.args.timeout)
        current_world = self.client.get_world()
        if self.args.skip_load_world or current_world.get_map().name.endswith(self.args.map):
            self.world = current_world
        else:
            self.world = self.client.load_world(self.args.map)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.args.fixed_delta_seconds
        settings.no_rendering_mode = self.args.no_rendering
        self.world.apply_settings(settings)

        bp = self.world.get_blueprint_library().filter(self.args.vehicle_type)[0]
        transform = carla.Transform(
            carla.Location(x=self.args.x, y=self.args.y, z=self.args.z + 2.0),
            carla.Rotation(roll=self.args.roll, pitch=self.args.pitch, yaw=self.args.yaw),
        )
        self.ego = self.world.spawn_actor(bp, transform)
        self.actors.append(self.ego)
        self.ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.ego.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.spectator = self.world.get_spectator()

        lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        attrs = {
            "range": "100",
            "channels": "64",
            "points_per_second": "300000",
            "upper_fov": "10.0",
            "lower_fov": "-30.0",
            "rotation_frequency": "20",
            "noise_stddev": "0.0",
        }
        for key, value in attrs.items():
            lidar_bp.set_attribute(key, value)

        # Matches autoware_carla_interface SensorKitLoader conversion:
        # velodyne_top_base_link x=-1.0 is translated from base_link to
        # CARLA vehicle center by subtracting wheelbase/2.
        lidar_transform = carla.Transform(
            carla.Location(x=-2.425, y=0.0, z=1.0),
            carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0),
        )
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego)
        self.actors.append(self.lidar)
        self.lidar.listen(self.on_lidar)

        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.ego
        )
        self.actors.append(self.collision_sensor)
        self.collision_sensor.listen(self.on_collision)

        # side output: screen captures for figures.
        # fully separate from the logging path; a failure here must not affect the drive.
        if getattr(self.args, "capture_frames", False):
            try:
                self._setup_capture()
            except Exception as exc:  # noqa: BLE001
                print(f"CAPTURE_SETUP_FAILED (run continues): {exc}", flush=True)

    def _setup_capture(self):
        import os
        os.makedirs(self.args.capture_dir, exist_ok=True)
        self._capture_warned = False

        def make_cam(loc, rot):
            bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", "1280")
            bp.set_attribute("image_size_y", "720")
            bp.set_attribute("sensor_tick", str(self.args.capture_interval))
            cam = self.world.spawn_actor(
                bp, carla.Transform(loc, rot), attach_to=self.ego)
            self.actors.append(cam)
            return cam

        win = None
        if getattr(self.args, "capture_window", ""):
            try:
                a, b = self.args.capture_window.split(":")
                win = (float(a), float(b))
            except Exception:  # noqa: BLE001
                print(f"CAPTURE_WINDOW_INVALID '{self.args.capture_window}' "
                      "(capturing the whole run)", flush=True)

        def saver(tag):
            def cb(image):
                try:
                    if win and not (win[0] <= image.timestamp <= win[1]):
                        return
                    image.save_to_disk(
                        f"{self.args.capture_dir}/{tag}_t{image.timestamp:09.2f}.png")
                except Exception as exc:  # noqa: BLE001
                    if not self._capture_warned:
                        self._capture_warned = True
                        print(f"CAPTURE_SAVE_FAILED (run continues): {exc}",
                              flush=True)
            return cb

        cam_chase = make_cam(carla.Location(x=-8.0, z=5.0),
                             carla.Rotation(pitch=-20.0))
        cam_chase.listen(saver("chase"))
        cam_bev = make_cam(carla.Location(z=50.0), carla.Rotation(pitch=-90.0))
        cam_bev.listen(saver("bev"))
        print(f"capture enabled -> {self.args.capture_dir} "
              f"(2 cams, {self.args.capture_interval}s)", flush=True)

    def update_spectator(self):
        if not self.args.spectator_follow or not self.ego or not self.ego.is_alive:
            return
        ego_tf = self.ego.get_transform()
        yaw = math.radians(ego_tf.rotation.yaw)
        forward = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)
        target = ego_tf.location + carla.Location(z=self.args.spectator_target_height)
        camera_location = carla.Location(
            x=target.x - forward.x * self.args.spectator_distance,
            y=target.y - forward.y * self.args.spectator_distance,
            z=target.z + self.args.spectator_height,
        )
        self.spectator.set_transform(
            carla.Transform(camera_location, look_at_rotation(camera_location, target))
        )

    def on_lidar(self, measurement):
        try:
            self.lidar_queue.put_nowait(measurement)
        except queue.Full:
            try:
                self.lidar_queue.get_nowait()
            except queue.Empty:
                pass
            self.lidar_queue.put_nowait(measurement)

    def on_collision(self, event):
        self.collision_detected = True
        self.node.get_logger().error(
            "collision detected: actor=%s impulse=(%.3f, %.3f, %.3f)"
            % (
                getattr(event.other_actor, "type_id", "unknown"),
                event.normal_impulse.x,
                event.normal_impulse.y,
                event.normal_impulse.z,
            )
        )

    def _run_short_motion(self, request, response):
        if self.motion_active:
            response.success = False
            response.message = "motion already active"
            return response
        if not self.ego or not self.ego.is_alive:
            response.success = False
            response.message = "ego actor is not alive"
            return response
        self.motion_active = True
        self.motion_done = False
        self.motion_start_time = None
        self.drive_start_time = None
        self.brake_start_time = None
        self.initial_location = self.ego.get_location()
        self.max_speed = 0.0
        self.max_distance = 0.0
        self.collision_detected = False
        self.current_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        response.success = True
        response.message = "short motion armed"
        self.node.get_logger().info("short motion trigger accepted")
        return response

    def _run_lane_route(self, request, response):
        if not self.route:
            response.success = False
            response.message = "route file not loaded"
            return response
        if self.motion_active or self.route_active:
            response.success = False
            response.message = "motion already active"
            return response
        if not self.ego or not self.ego.is_alive:
            response.success = False
            response.message = "ego actor is not alive"
            return response
        self.route_active = True
        self.route_done = False
        self.route_phase = "route"
        self.route_trigger_time = None
        self.route_brake_start_time = None
        self.route_abort_reason = ""
        self.route_prev_progress = 0.0
        self.route_last_progress_time = None
        self.route_lateral_exceed_since = None
        self.route_speed_integral = 0.0
        self.route_prev_speed_error = 0.0
        self.route_prev_control_time = None
        self.route_v_ref = 0.0
        self.route_speed_error = 0.0
        self.collision_detected = False
        self.initial_location = self.ego.get_location()
        self.max_speed = 0.0
        self.max_distance = 0.0
        self.current_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        response.success = True
        response.message = "lane route armed"
        self.node.get_logger().info(
            "lane route trigger accepted: waypoints=%d length=%.3f"
            % (len(self.route), self.route[-1].get("s", 0.0))
        )
        return response

    def _speed_and_distance(self):
        vel = self.ego.get_velocity()
        speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
        loc = self.ego.get_location()
        if self.initial_location is None:
            distance = 0.0
        else:
            dx = loc.x - self.initial_location.x
            dy = loc.y - self.initial_location.y
            distance = math.sqrt(dx * dx + dy * dy)
        self.max_speed = max(self.max_speed, speed)
        self.max_distance = max(self.max_distance, distance)
        return speed, distance

    def _set_brake(self):
        self.current_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

    def _route_nearest_index(self, loc):
        best_i = 0
        best_d2 = float("inf")
        for i, wp in enumerate(self.route):
            dx = loc.x - wp["x"]
            dy = loc.y - wp["y"]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_i = i
                best_d2 = d2
        return best_i, math.sqrt(best_d2)

    def _route_target_index(self, current_index, lookahead_m):
        current_s = self.route[current_index]["s"]
        target_s = current_s + lookahead_m
        for i in range(current_index, len(self.route)):
            if self.route[i]["s"] >= target_s:
                return i
        return len(self.route) - 1

    def _route_metrics(self):
        ego_tf = self.ego.get_transform()
        loc = ego_tf.location
        yaw = ego_tf.rotation.yaw
        idx, _ = self._route_nearest_index(loc)
        wp = self.route[idx]
        route_yaw = wp["yaw"]
        dx = loc.x - wp["x"]
        dy = loc.y - wp["y"]
        right_x = -math.sin(math.radians(route_yaw))
        right_y = math.cos(math.radians(route_yaw))
        lateral = dx * right_x + dy * right_y
        heading_error = angle_diff_deg(route_yaw, yaw)
        progress = wp["s"]
        return idx, progress, lateral, heading_error

    def _route_has_speed_profile(self):
        return self.args.route_use_speed_profile or any("v_ref_mps" in wp for wp in self.route)

    def _route_v_ref_at(self, idx, progress, route_length):
        if not self._route_has_speed_profile():
            return self.args.route_target_speed
        idx = max(0, min(idx, len(self.route) - 1))
        wp = self.route[idx]
        if "v_ref_mps" in wp:
            v_ref = float(wp["v_ref_mps"])
        else:
            v_ref = self.args.route_target_speed
        # Keep the final stopping behavior progress-based even if the route
        # file was generated with sparse speed annotations.
        remaining = max(0.0, route_length - progress)
        if remaining < self.args.route_stop_distance:
            scale = max(0.0, remaining / max(1e-6, self.args.route_stop_distance))
            v_ref = min(v_ref, self.args.route_stop_speed + scale * (v_ref - self.args.route_stop_speed))
        return max(0.0, min(self.args.route_profile_speed_cap, v_ref))

    def _speed_feedback_control(self, timestamp, speed, v_ref):
        dt = self.args.fixed_delta_seconds
        if self.route_prev_control_time is not None:
            dt = max(1e-3, timestamp - self.route_prev_control_time)
        self.route_prev_control_time = timestamp

        error = v_ref - speed
        self.route_speed_integral += error * dt
        lim = self.args.route_speed_integral_limit
        self.route_speed_integral = max(-lim, min(lim, self.route_speed_integral))
        d_error = (error - self.route_prev_speed_error) / dt
        self.route_prev_speed_error = error

        accel_cmd = (
            self.args.route_speed_kp * error
            + self.args.route_speed_ki * self.route_speed_integral
            + self.args.route_speed_kd * d_error
        )
        if accel_cmd >= 0.0:
            throttle = min(self.args.route_max_throttle_cmd, accel_cmd)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(self.args.route_max_brake_cmd, -accel_cmd * self.args.route_brake_gain)
        return throttle, brake, error

    def _publish_route_state(self, timestamp):
        if not self.route:
            return
        msg = String()
        msg.data = json.dumps(self.route_state, sort_keys=True)
        self.route_state_pub.publish(msg)

    def update_lane_route_control(self, timestamp):
        if not self.route or not self.route_active:
            return False
        if not self.ego or not self.ego.is_alive:
            self.route_abort_reason = "ego_not_alive"
            self.route_phase = "brake"
            self.route_brake_start_time = self.route_brake_start_time or timestamp
            self._set_brake()
            return True

        if self.route_trigger_time is None:
            self.route_trigger_time = timestamp
            self.route_last_progress_time = timestamp

        speed, distance_from_start = self._speed_and_distance()
        idx, progress, lateral, heading_error = self._route_metrics()

        if progress > self.route_prev_progress + 0.2:
            self.route_prev_progress = progress
            self.route_last_progress_time = timestamp

        abort = ""
        if self.collision_detected:
            abort = "collision"
        elif speed > self.args.max_speed:
            abort = "max_speed"
        elif distance_from_start > self.args.route_corridor_distance:
            abort = "route_corridor_distance"
        elif abs(lateral) > self.args.route_max_lateral_deviation:
            if self.route_lateral_exceed_since is None:
                self.route_lateral_exceed_since = timestamp
            elif timestamp - self.route_lateral_exceed_since >= self.args.route_lateral_abort_sec:
                abort = "lane_deviation"
        else:
            self.route_lateral_exceed_since = None

        if (
            not abort
            and self.route_brake_start_time is None
            and self.route_last_progress_time is not None
            and timestamp - self.route_last_progress_time > self.args.route_progress_timeout
        ):
            abort = "progress_timeout"

        route_length = self.route[-1]["s"]
        complete = progress >= route_length - self.args.route_goal_tolerance
        v_ref = 0.0
        speed_error = 0.0
        speed_profile_enabled = self._route_has_speed_profile()

        if abort:
            if not self.route_abort_reason:
                self.route_abort_reason = abort
                self.node.get_logger().error(
                    "lane route hard stop: reason=%s progress=%.3f lateral=%.3f speed=%.3f"
                    % (abort, progress, lateral, speed)
                )
            self.route_phase = "brake"
            self.route_brake_start_time = self.route_brake_start_time or timestamp
            self._set_brake()
        elif complete:
            if self.route_brake_start_time is None:
                self.node.get_logger().info(
                    "lane route brake phase started: progress=%.3f speed=%.3f lateral=%.3f"
                    % (progress, speed, lateral)
                )
                self.route_brake_start_time = timestamp
            self.route_phase = "brake"
            self._set_brake()
        elif timestamp - self.route_trigger_time < self.args.motion_pre_hold:
            self.route_phase = "before"
            self._set_brake()
        else:
            self.route_phase = "route"
            target_i = self._route_target_index(idx, self.args.route_lookahead)
            target = self.route[target_i]
            ego_tf = self.ego.get_transform()
            yaw = math.radians(ego_tf.rotation.yaw)
            dx = target["x"] - ego_tf.location.x
            dy = target["y"] - ego_tf.location.y
            local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
            steer = max(
                -self.args.route_max_steer,
                min(self.args.route_max_steer, self.args.route_steer_gain * local_y),
            )
            v_ref = self._route_v_ref_at(idx, progress, route_length)
            if speed_profile_enabled:
                throttle, brake, speed_error = self._speed_feedback_control(timestamp, speed, v_ref)
            else:
                speed_error = self.args.route_target_speed - speed
                throttle = self.args.route_throttle
                brake = 0.0
                if speed >= self.args.route_target_speed:
                    throttle = 0.0
                if speed > self.args.route_target_speed + 0.25:
                    brake = 0.15
            self.current_control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

        if self.route_brake_start_time is not None:
            v_ref = 0.0
            speed_error = -speed
            self._set_brake()
            if timestamp - self.route_brake_start_time >= self.args.motion_post_hold:
                self.route_active = False
                self.route_done = True
                self.route_phase = "after"
                status = "FAIL" if self.route_abort_reason else "PASS"
                self.node.get_logger().info(
                    "lane route completed: status=%s progress=%.3f max_speed=%.3f max_distance=%.3f"
                    % (status, progress, self.max_speed, self.max_distance)
                )

        self.ego.apply_control(self.current_control)
        self.route_v_ref = v_ref
        self.route_speed_error = speed_error
        self.route_state = {
            "stamp": timestamp,
            "phase": self.route_phase,
            "active": self.route_active,
            "done": self.route_done,
            "abort_reason": self.route_abort_reason,
            "route_progress": progress,
            "route_length": route_length,
            "waypoint_index": idx,
            "ego_x": self.ego.get_transform().location.x,
            "ego_y": self.ego.get_transform().location.y,
            "ego_z": self.ego.get_transform().location.z,
            "ego_yaw": self.ego.get_transform().rotation.yaw,
            "ego_speed": speed,
            "lane_lateral_deviation": lateral,
            "route_heading_error": heading_error,
            "v_ref_mps": v_ref,
            "speed_error_mps": speed_error,
            "speed_profile_enabled": bool(speed_profile_enabled),
            "throttle": float(self.current_control.throttle),
            "brake": float(self.current_control.brake),
            "steer": float(self.current_control.steer),
            "collision": bool(self.collision_detected),
            "safety_abort": bool(self.route_abort_reason),
        }
        return True

    def update_control(self, timestamp):
        if not self.ego or not self.ego.is_alive:
            self._set_brake()
            return
        if self.update_lane_route_control(timestamp):
            return
        if not self.motion_active:
            self._set_brake()
            return

        if self.motion_start_time is None:
            self.motion_start_time = timestamp

        speed, distance = self._speed_and_distance()
        elapsed = timestamp - self.motion_start_time

        hard_stop_reason = None
        if self.collision_detected:
            hard_stop_reason = "collision"
        elif speed > self.args.max_speed:
            hard_stop_reason = "max_speed"
        elif distance > self.args.safety_distance:
            hard_stop_reason = "safety_distance"

        if hard_stop_reason:
            if self.brake_start_time is None:
                self.brake_start_time = timestamp
                self.node.get_logger().error(
                    "short motion hard stop: reason=%s distance=%.3f speed=%.3f"
                    % (hard_stop_reason, distance, speed)
                )
            self._set_brake()
        elif elapsed < self.args.motion_pre_hold:
            self._set_brake()
        elif self.drive_start_time is None:
            self.drive_start_time = timestamp
            self.current_control = carla.VehicleControl(
                throttle=self.args.motion_throttle, steer=0.0, brake=0.0
            )
            self.node.get_logger().info("short motion drive phase started")
        elif (
            distance >= self.args.motion_max_distance
            or timestamp - self.drive_start_time >= self.args.motion_max_drive_sec
        ):
            if self.brake_start_time is None:
                self.brake_start_time = timestamp
                self.node.get_logger().info(
                    "short motion brake phase started: distance=%.3f speed=%.3f"
                    % (distance, speed)
                )
            self._set_brake()
        else:
            throttle = self.args.motion_throttle
            if self.args.motion_cruise_speed > 0.0 and speed >= self.args.motion_cruise_speed:
                throttle = 0.0
            self.current_control = carla.VehicleControl(
                throttle=throttle, steer=0.0, brake=0.0
            )

        if self.brake_start_time is not None:
            self._set_brake()
            if timestamp - self.brake_start_time >= self.args.motion_post_hold:
                self.motion_active = False
                self.motion_done = True
                self.node.get_logger().info(
                    "short motion completed: max_distance=%.3f max_speed=%.3f"
                    % (self.max_distance, self.max_speed)
                )

        self.ego.apply_control(self.current_control)

    def publish_status(self, stamp, timestamp):
        clock = Clock()
        clock.clock = stamp
        self.clock_pub.publish(clock)

        ego_tf = self.ego.get_transform()
        pose = carla_transform_to_ros_pose(ego_tf)
        pose.header.stamp = stamp
        self.pose_pub.publish(pose)

        vel = self.ego.get_velocity()
        yaw = math.radians(ego_tf.rotation.yaw)
        longitudinal = math.cos(yaw) * vel.x + math.sin(yaw) * vel.y
        lateral = -math.sin(yaw) * vel.x + math.cos(yaw) * vel.y
        vel_msg = VelocityReport()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = "base_link"
        vel_msg.longitudinal_velocity = longitudinal
        vel_msg.lateral_velocity = lateral
        vel_msg.heading_rate = 0.0
        self.vel_pub.publish(vel_msg)

        cmd_msg = ActuationCommandStamped()
        cmd_msg.header.stamp = stamp
        cmd_msg.header.frame_id = "base_link"
        cmd_msg.actuation.accel_cmd = float(self.current_control.throttle)
        cmd_msg.actuation.brake_cmd = float(self.current_control.brake)
        cmd_msg.actuation.steer_cmd = float(self.current_control.steer)
        self.control_pub.publish(cmd_msg)
        if self.route:
            if not self.route_state:
                self.route_state = {
                    "stamp": timestamp,
                    "phase": self.route_phase,
                    "active": False,
                    "done": self.route_done,
                    "abort_reason": self.route_abort_reason,
                    "route_progress": 0.0,
                    "route_length": self.route[-1]["s"],
                    "waypoint_index": 0,
                    "ego_x": ego_tf.location.x,
                    "ego_y": ego_tf.location.y,
                    "ego_z": ego_tf.location.z,
                    "ego_yaw": ego_tf.rotation.yaw,
                    "ego_speed": math.hypot(vel.x, vel.y),
                    "lane_lateral_deviation": math.nan,
                    "route_heading_error": math.nan,
                    "v_ref_mps": self.route_v_ref,
                    "speed_error_mps": self.route_speed_error,
                    "speed_profile_enabled": bool(self._route_has_speed_profile()),
                    "throttle": float(self.current_control.throttle),
                    "brake": float(self.current_control.brake),
                    "steer": float(self.current_control.steer),
                    "collision": bool(self.collision_detected),
                    "safety_abort": bool(self.route_abort_reason),
                }
            self._publish_route_state(timestamp)

    def run(self):
        self.setup()
        self.node.get_logger().info(
            "minimal bridge running: spawn=(%.3f, %.3f, %.3f yaw %.1f), no_rendering=%s"
            % (self.args.x, self.args.y, self.args.z, self.args.yaw, self.args.no_rendering)
        )
        try:
            while rclpy.ok():
                frame = self.world.tick()
                snapshot = self.world.get_snapshot()
                timestamp = snapshot.timestamp.elapsed_seconds
                stamp = ros_stamp(timestamp)
                self.update_control(timestamp)
                self.update_spectator()
                self.publish_status(stamp, timestamp)
                while True:
                    try:
                        measurement = self.lidar_queue.get_nowait()
                    except queue.Empty:
                        break
                    self.lidar_pub.publish(lidar_to_pointcloud2(measurement, stamp))
                rclpy.spin_once(self.node, timeout_sec=0.0)
                if self.args.wall_sleep > 0:
                    time.sleep(self.args.wall_sleep)
        finally:
            for actor in reversed(self.actors):
                if actor.is_alive:
                    actor.destroy()
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            self.node.destroy_node()
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--map", default="Town01")
    parser.add_argument("--skip-load-world", action="store_true")
    parser.add_argument("--vehicle-type", default="vehicle.toyota.prius")
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--wall-sleep", type=float, default=0.0)
    parser.add_argument("--motion-throttle", type=float, default=0.25)
    parser.add_argument("--motion-cruise-speed", type=float, default=0.0)
    parser.add_argument("--motion-pre-hold", type=float, default=1.0)
    parser.add_argument("--motion-max-drive-sec", type=float, default=3.0)
    parser.add_argument("--motion-max-distance", type=float, default=8.0)
    parser.add_argument("--motion-post-hold", type=float, default=8.0)
    parser.add_argument("--max-speed", type=float, default=2.0)
    parser.add_argument("--safety-distance", type=float, default=10.0)
    parser.add_argument("--spectator-follow", action="store_true")
    parser.add_argument("--spectator-distance", type=float, default=8.0)
    parser.add_argument("--spectator-height", type=float, default=4.0)
    parser.add_argument("--spectator-target-height", type=float, default=0.8)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, default=0.3)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--route-file", default="")
    parser.add_argument("--capture-frames", action="store_true",
                        help="screen captures for figures (side output)")
    parser.add_argument("--capture-dir", default="")
    parser.add_argument("--capture-interval", type=float, default=0.5)
    parser.add_argument("--capture-window", default="",
                        help="'t0:t1' simulation-second window to capture (empty = whole run)")
    parser.add_argument("--route-use-speed-profile", action="store_true")
    parser.add_argument("--route-throttle", type=float, default=0.20)
    parser.add_argument("--route-target-speed", type=float, default=1.45)
    parser.add_argument("--route-profile-speed-cap", type=float, default=5.5)
    parser.add_argument("--route-stop-distance", type=float, default=45.0)
    parser.add_argument("--route-stop-speed", type=float, default=0.4)
    parser.add_argument("--route-speed-kp", type=float, default=0.24)
    parser.add_argument("--route-speed-ki", type=float, default=0.035)
    parser.add_argument("--route-speed-kd", type=float, default=0.0)
    parser.add_argument("--route-speed-integral-limit", type=float, default=4.0)
    parser.add_argument("--route-max-throttle-cmd", type=float, default=0.45)
    parser.add_argument("--route-max-brake-cmd", type=float, default=0.70)
    parser.add_argument("--route-brake-gain", type=float, default=0.55)
    parser.add_argument("--route-lookahead", type=float, default=5.0)
    parser.add_argument("--route-steer-gain", type=float, default=0.12)
    parser.add_argument("--route-max-steer", type=float, default=0.35)
    parser.add_argument("--route-goal-tolerance", type=float, default=1.5)
    parser.add_argument("--route-max-lateral-deviation", type=float, default=1.2)
    parser.add_argument("--route-lateral-abort-sec", type=float, default=0.5)
    parser.add_argument("--route-progress-timeout", type=float, default=5.0)
    parser.add_argument("--route-corridor-distance", type=float, default=130.0)
    MinimalCarlaBridge(parser.parse_args()).run()


if __name__ == "__main__":
    main()
