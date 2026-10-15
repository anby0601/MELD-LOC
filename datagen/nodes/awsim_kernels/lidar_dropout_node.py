#!/usr/bin/env python3
"""
LiDAR point perturbation node for AWSIM failure scenarios.

Subscribes to a PointCloud2 topic, applies a time-scheduled perturbation, and
republishes the perturbed cloud on a separate topic. Header (stamp / frame_id)
is preserved so downstream NDT timing is unchanged.

Supported modes:
  - dropout: randomly drop a fraction of points.
  - range_clip: keep points whose range is <= scheduled radius [m].
  - sector_remove: remove a scheduled horizontal azimuth sector [deg].
  - yaw_rotate: rotate every point about the sensor z-axis by a scheduled angle
    [deg] (proxy for a LiDAR extrinsic yaw bias). No points are dropped; x/y are
    rotated about (0,0), z/intensity/ring/distance are preserved. The `azimuth`
    field (if present) is left unchanged, which for small angles (<= ~2 deg) is a
    negligible inconsistency relative to the injected x/y rotation.
  - translate_y: shift every point by a scheduled offset [m] along the sensor +y
    (lateral) axis (proxy for a LiDAR lateral extrinsic/mounting bias). No points
    are dropped; only y is changed, x/z/intensity/ring preserved. Intended to
    surface cross-track (lateral) pose error for clean lateral labels.

Design notes:
  - Schedule timing aligns to drive start, not node start. Either a trigger
    topic (first message starts the clock) or a fixed start-delay is used.
  - For every scan we log perturbation target, input/output counts, and actual
    keep/removed ratios into a CSV (--log-csv).
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from sensor_msgs.msg import PointCloud2, PointField


SCHEDULE_TYPE = List[Tuple[float, float]]


POINT_FIELD_DTYPES = {
    PointField.INT8: ("i1", 1),
    PointField.UINT8: ("u1", 1),
    PointField.INT16: ("i2", 2),
    PointField.UINT16: ("u2", 2),
    PointField.INT32: ("i4", 4),
    PointField.UINT32: ("u4", 4),
    PointField.FLOAT32: ("f4", 4),
    PointField.FLOAT64: ("f8", 8),
}


def parse_schedule(spec: str, mode: str) -> SCHEDULE_TYPE:
    out = []
    for pair in spec.split(","):
        t_str, value_str = pair.split(":")
        t = float(t_str)
        value = float(value_str)
        if mode == "dropout" and not (0.0 <= value <= 1.0):
            raise ValueError(f"dropout ratio out of range: {value}")
        if mode == "range_clip" and not (value > 0.0 or math.isinf(value)):
            raise ValueError(f"range_clip radius must be positive or inf: {value}")
        if mode == "sector_remove" and not (0.0 <= value <= 360.0):
            raise ValueError(f"sector_remove width must be in [0, 360] deg: {value}")
        if mode == "sector_thin" and not (0.0 <= value <= 1.0):
            raise ValueError(f"sector_thin remove-ratio must be in [0, 1]: {value}")
        if mode == "yaw_rotate" and not (-45.0 <= value <= 45.0):
            raise ValueError(f"yaw_rotate angle must be in [-45, 45] deg: {value}")
        if mode == "translate_y" and not (-3.0 <= value <= 3.0):
            raise ValueError(f"translate_y offset must be in [-3, 3] m: {value}")
        out.append((t, value))
    out.sort(key=lambda x: x[0])
    return out


def schedule_value_at(schedule: SCHEDULE_TYPE, t_rel: float, default_value: float) -> float:
    """Step-wise lookup: value = last schedule entry with t<=t_rel."""
    if t_rel < schedule[0][0]:
        return default_value
    cur = default_value
    for t_start, value in schedule:
        if t_rel >= t_start:
            cur = value
        else:
            break
    return cur


def _field_dtype(field: PointField, is_bigendian: bool):
    if field.datatype not in POINT_FIELD_DTYPES:
        raise ValueError(f"unsupported PointField datatype for {field.name}: {field.datatype}")
    code, size = POINT_FIELD_DTYPES[field.datatype]
    endian = ">" if is_bigendian and size > 1 else "<"
    return np.dtype(endian + code)


def _get_numeric_field(msg: PointCloud2, field_name: str, raw_points: np.ndarray):
    field = next((f for f in msg.fields if f.name == field_name), None)
    if field is None:
        return None
    dtype = _field_dtype(field, msg.is_bigendian)
    if field.offset + dtype.itemsize > int(msg.point_step):
        raise ValueError(
            f"field {field_name} offset {field.offset} exceeds point_step {msg.point_step}"
        )
    return np.ndarray(
        shape=(raw_points.shape[0],),
        dtype=dtype,
        buffer=raw_points,
        offset=field.offset,
        strides=(int(msg.point_step),),
    ).astype(np.float64, copy=False)


def keep_indices_for_dropout(n_in: int, ratio: float, rng: np.random.Generator):
    if n_in == 0 or ratio <= 0.0:
        return np.arange(n_in, dtype=np.int64)
    n_keep = max(0, int(round(n_in * (1.0 - ratio))))
    if n_keep >= n_in:
        return np.arange(n_in, dtype=np.int64)
    keep = rng.choice(n_in, size=n_keep, replace=False)
    keep.sort()
    return keep


def keep_indices_for_range_clip(msg: PointCloud2, raw_points: np.ndarray, radius_m: float):
    if raw_points.shape[0] == 0 or math.isinf(radius_m):
        return np.arange(raw_points.shape[0], dtype=np.int64)

    distance = _get_numeric_field(msg, "distance", raw_points)
    if distance is None:
        x = _get_numeric_field(msg, "x", raw_points)
        y = _get_numeric_field(msg, "y", raw_points)
        z = _get_numeric_field(msg, "z", raw_points)
        if x is None or y is None or z is None:
            raise ValueError("range_clip requires either a distance field or x/y/z fields")
        distance = np.sqrt(x * x + y * y + z * z)

    mask = np.isfinite(distance) & (distance <= radius_m)
    return np.flatnonzero(mask).astype(np.int64)


def sector_center_deg(direction: str) -> float:
    if direction == "front":
        return 0.0
    if direction == "left":
        return 90.0
    if direction == "right":
        return -90.0
    raise ValueError(f"unsupported sector direction: {direction}")


def _wrap_deg(angle_deg: np.ndarray):
    return (angle_deg + 180.0) % 360.0 - 180.0


def _azimuth_deg(msg: PointCloud2, raw_points: np.ndarray):
    azimuth = _get_numeric_field(msg, "azimuth", raw_points)
    if azimuth is not None:
        finite = azimuth[np.isfinite(azimuth)]
        if finite.size and np.nanmax(np.abs(finite)) <= (2.0 * math.pi + 1e-3):
            return np.degrees(azimuth)
        return azimuth

    x = _get_numeric_field(msg, "x", raw_points)
    y = _get_numeric_field(msg, "y", raw_points)
    if x is None or y is None:
        raise ValueError("sector_remove requires either an azimuth field or x/y fields")
    return np.degrees(np.arctan2(y, x))


def keep_indices_for_sector_remove(
    msg: PointCloud2,
    raw_points: np.ndarray,
    center_deg: float,
    width_deg: float,
):
    n_in = raw_points.shape[0]
    if n_in == 0 or width_deg <= 0.0:
        return np.arange(n_in, dtype=np.int64)
    if width_deg >= 360.0:
        return np.empty((0,), dtype=np.int64)

    azimuth_deg = _azimuth_deg(msg, raw_points)
    diff = np.abs(_wrap_deg(azimuth_deg - center_deg))
    remove = np.isfinite(diff) & (diff <= (width_deg / 2.0))
    return np.flatnonzero(~remove).astype(np.int64)


def keep_indices_for_sector_thin(
    msg: PointCloud2,
    raw_points: np.ndarray,
    center_deg: float,
    width_deg: float,
    remove_ratio: float,
    rng: np.random.Generator,
):
    """Probabilistic sector thinning: within the [center +/- width/2] sector,
    randomly drop `remove_ratio` of the points (instead of removing the whole
    wedge). Points outside the sector are always kept. This weakens directional
    observability *gradually* rather than the all-or-nothing wedge removal."""
    n_in = raw_points.shape[0]
    if n_in == 0 or remove_ratio <= 0.0 or width_deg <= 0.0:
        return np.arange(n_in, dtype=np.int64)
    azimuth_deg = _azimuth_deg(msg, raw_points)
    diff = np.abs(_wrap_deg(azimuth_deg - center_deg))
    in_sector = np.flatnonzero(np.isfinite(diff) & (diff <= (width_deg / 2.0)))
    if in_sector.size == 0:
        return np.arange(n_in, dtype=np.int64)
    n_remove = int(round(in_sector.size * min(1.0, remove_ratio)))
    if n_remove <= 0:
        return np.arange(n_in, dtype=np.int64)
    remove_sel = rng.choice(in_sector, size=min(n_remove, in_sector.size), replace=False)
    remove_mask = np.zeros(n_in, dtype=bool)
    remove_mask[remove_sel] = True
    return np.flatnonzero(~remove_mask).astype(np.int64)


def apply_yaw_rotation(msg: PointCloud2, raw_points: np.ndarray, yaw_deg: float) -> bytes:
    """Rotate every point about the sensor z-axis by yaw_deg [deg].

    Keeps all points (n_out == n_in). Only the x/y fields are rewritten; z,
    intensity, ring, distance and every other field are preserved byte-for-byte.
    Range is invariant under a z-rotation so the `distance` field stays correct;
    the `azimuth` field is intentionally left unchanged (small-angle proxy).
    Returns the new point data as bytes.
    """
    n = raw_points.shape[0]
    if n == 0 or yaw_deg == 0.0:
        return raw_points.tobytes()

    x = _get_numeric_field(msg, "x", raw_points)
    y = _get_numeric_field(msg, "y", raw_points)
    if x is None or y is None:
        raise ValueError("yaw_rotate requires x/y fields")

    rad = math.radians(yaw_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    xr = x * c - y * s
    yr = x * s + y * c

    # writable copy of the raw byte buffer; rewrite x/y in place at field strides
    buf = bytearray(raw_points.tobytes())
    step = int(msg.point_step)
    x_field = next(f for f in msg.fields if f.name == "x")
    y_field = next(f for f in msg.fields if f.name == "y")
    x_dtype = _field_dtype(x_field, msg.is_bigendian)
    y_dtype = _field_dtype(y_field, msg.is_bigendian)
    x_view = np.ndarray(shape=(n,), dtype=x_dtype, buffer=buf,
                        offset=int(x_field.offset), strides=(step,))
    y_view = np.ndarray(shape=(n,), dtype=y_dtype, buffer=buf,
                        offset=int(y_field.offset), strides=(step,))
    x_view[:] = xr.astype(x_dtype)
    y_view[:] = yr.astype(y_dtype)
    return bytes(buf)


def apply_lateral_translation(msg: PointCloud2, raw_points: np.ndarray, offset_m: float) -> bytes:
    """Shift every point by offset_m along the sensor +y (lateral) axis.

    Proxy for a LiDAR lateral extrinsic/mounting bias. Keeps all points
    (n_out == n_in); only the y field is rewritten, x/z/intensity/ring and every
    other field are preserved byte-for-byte. Unlike a z-rotation this has no
    rotational symmetry, so the residual is expected to surface as cross-track
    (lateral) pose error rather than along-track slip. Returns new point bytes.
    """
    n = raw_points.shape[0]
    if n == 0 or offset_m == 0.0:
        return raw_points.tobytes()

    buf = bytearray(raw_points.tobytes())
    step = int(msg.point_step)
    y_field = next(f for f in msg.fields if f.name == "y")
    y_dtype = _field_dtype(y_field, msg.is_bigendian)
    y_view = np.ndarray(shape=(n,), dtype=y_dtype, buffer=buf,
                        offset=int(y_field.offset), strides=(step,))
    y_view[:] = (y_view.astype(np.float64) + offset_m).astype(y_dtype)
    return bytes(buf)


class LidarDropoutNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("lidar_dropout_node")
        self._args = args
        self._schedule = parse_schedule(args.schedule, args.mode)
        self._rng = np.random.default_rng(args.seed)
        self._log_rows = []
        self._log_path = Path(args.log_csv) if args.log_csv else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # trigger state
        self._t0_ns = None  # nanosecond reference (drive-start time, picked from ROS clock)
        self._started = False

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._pub = self.create_publisher(PointCloud2, args.output_topic, qos_be)

        # Subscribe to LiDAR (perturbation pipeline).
        self.create_subscription(PointCloud2, args.input_topic, self._cb_cloud, qos_be)

        # Trigger setup
        if args.start_trigger_topic:
            # We don't know the trigger msg type up front; many possible topics are simple.
            # Use generic AnyMsg via rclpy? rclpy does not have AnyMsg directly. Instead,
            # subscribe via a type loader if known. For simplicity, default trigger uses
            # autoware_control_msgs/Control (matches drive_waypoints.py first publish).
            from autoware_control_msgs.msg import Control
            qos_reliable = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(Control, args.start_trigger_topic, self._cb_trigger,
                                     qos_reliable)
            self.get_logger().info(
                f"Waiting for first message on trigger topic {args.start_trigger_topic} "
                f"(then schedule t=0)."
            )
        else:
            # immediate start (still respect start_delay_sec)
            self._t0_ns = self.get_clock().now().nanoseconds + int(args.start_delay_sec * 1e9)
            self.get_logger().info(
                f"No trigger topic. Schedule starts {args.start_delay_sec:.1f} s from now."
            )

        self.get_logger().info(
            f"LidarDropoutNode ready. mode={args.mode} input={args.input_topic} "
            f"output={args.output_topic}\n"
            f"schedule={self._schedule}  seed={args.seed}  start_delay={args.start_delay_sec}s"
        )

    def _cb_trigger(self, _msg):
        if self._t0_ns is None:
            self._t0_ns = self.get_clock().now().nanoseconds + int(self._args.start_delay_sec * 1e9)
            self.get_logger().info(
                f"Trigger received. Schedule starts in {self._args.start_delay_sec:.1f} s "
                f"(at clock ns={self._t0_ns})."
            )

    def _cb_cloud(self, msg: PointCloud2):
        # always forward header/frame meta even before start
        now_ns = self.get_clock().now().nanoseconds
        if self._t0_ns is None or now_ns < self._t0_ns:
            # pre-start: forward as-is.
            t_rel = -1.0
            target_value = 0.0 if self._args.mode in ("dropout", "sector_remove", "sector_thin", "yaw_rotate", "translate_y") else math.inf
        else:
            t_rel = (now_ns - self._t0_ns) / 1e9
            target_value = schedule_value_at(
                self._schedule,
                t_rel,
                0.0 if self._args.mode in ("dropout", "sector_remove", "sector_thin", "yaw_rotate", "translate_y") else math.inf,
            )

        n_in = int(msg.width) * int(msg.height)
        step = int(msg.point_step)
        raw = bytes(msg.data)
        if n_in == 0:
            raw_points = np.empty((0, step), dtype=np.uint8)
        else:
            raw_points = np.frombuffer(raw, dtype=np.uint8).reshape(-1, step)

        target_ratio = math.nan
        target_radius_m = math.nan
        sector_width_deg = math.nan
        yaw_rotation_deg = math.nan
        rotation_center = ""
        lateral_offset_m = math.nan
        keep = None
        new_data = None
        t_perf0 = time.perf_counter()

        if self._args.mode == "dropout":
            keep = keep_indices_for_dropout(n_in, target_value, self._rng)
            target_ratio = target_value
        elif self._args.mode == "range_clip":
            keep = keep_indices_for_range_clip(msg, raw_points, target_value)
            target_radius_m = target_value
        elif self._args.mode == "sector_remove":
            keep = keep_indices_for_sector_remove(
                msg,
                raw_points,
                self._args.sector_center_deg,
                target_value,
            )
            sector_width_deg = target_value
        elif self._args.mode == "sector_thin":
            keep = keep_indices_for_sector_thin(
                msg,
                raw_points,
                self._args.sector_center_deg,
                self._args.sector_thin_width,
                target_value,
                self._rng,
            )
            target_ratio = target_value
            sector_width_deg = self._args.sector_thin_width
        elif self._args.mode == "yaw_rotate":
            # rotation keeps every point; transform x/y about sensor origin
            new_data = apply_yaw_rotation(msg, raw_points, target_value)
            n_out = n_in
            yaw_rotation_deg = target_value
            rotation_center = "0.0,0.0"
        elif self._args.mode == "translate_y":
            # lateral shift keeps every point; transform y only
            new_data = apply_lateral_translation(msg, raw_points, target_value)
            n_out = n_in
            lateral_offset_m = target_value
        else:
            raise RuntimeError(f"unsupported mode: {self._args.mode}")

        if keep is not None:
            n_out = int(keep.size)
            # build new pointcloud bytes (subset selection)
            new_data = msg.data if n_out == n_in else raw_points[keep].tobytes()

        elapsed_ms = (time.perf_counter() - t_perf0) * 1e3
        actual_keep_ratio = (n_out / n_in) if n_in else 0.0
        actual_removed_ratio = 1.0 - actual_keep_ratio if n_in else 0.0

        out = PointCloud2()
        out.header = msg.header  # preserve stamp + frame_id exactly
        out.height = 1
        out.width = n_out
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = msg.point_step * n_out
        out.is_dense = msg.is_dense
        out.data = bytes(new_data)

        self._pub.publish(out)

        # log
        row = {
            "t_rel": round(t_rel, 4),
            "mode": self._args.mode,
            "target_ratio": round(target_ratio, 4) if math.isfinite(target_ratio) else "",
            "target_radius_m": round(target_radius_m, 4) if math.isfinite(target_radius_m) else "",
            "sector_direction": self._args.sector_direction if self._args.mode == "sector_remove" else "",
            "sector_center_deg": round(self._args.sector_center_deg, 4) if self._args.mode == "sector_remove" else "",
            "sector_width_deg": round(sector_width_deg, 4) if math.isfinite(sector_width_deg) else "",
            "yaw_rotation_deg": round(yaw_rotation_deg, 4) if math.isfinite(yaw_rotation_deg) else "",
            "rotation_center": rotation_center,
            "lateral_offset_m": round(lateral_offset_m, 4) if math.isfinite(lateral_offset_m) else "",
            "target_value": round(target_value, 4) if math.isfinite(target_value) else "inf",
            "n_in": n_in,
            "n_out": n_out,
            "elapsed_ms": round(elapsed_ms, 4),
            # Backward-compatible alias: historically this meant removed ratio.
            "actual_ratio": round(actual_removed_ratio, 6),
            "actual_keep_ratio": round(actual_keep_ratio, 6),
            "actual_removed_ratio": round(actual_removed_ratio, 6),
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nsec": int(msg.header.stamp.nanosec),
        }
        self._log_rows.append(row)

        # periodic stdout log every 1 s of t_rel
        if self._args.verbose and t_rel >= 0.0:
            bucket = int(t_rel)
            if bucket != getattr(self, "_last_log_bucket", -1):
                self._last_log_bucket = bucket
                target_desc = (
                    f"drop={target_value*100:5.1f}%"
                    if self._args.mode == "dropout"
                    else (
                        f"radius={target_value:6.1f}m"
                        if self._args.mode == "range_clip"
                        else (
                            f"yaw_rot={target_value:+5.2f}deg"
                            if self._args.mode == "yaw_rotate"
                            else (
                                f"sector={self._args.sector_direction} "
                                f"center={self._args.sector_center_deg:5.1f}deg "
                                f"width={target_value:5.1f}deg"
                            )
                        )
                    )
                )
                self.get_logger().info(
                    f"t_rel={t_rel:6.2f}s  mode={self._args.mode}  {target_desc}  "
                    f"n_in={n_in}  n_out={n_out}  removed={actual_removed_ratio*100:5.2f}%"
                )

    def flush_log(self):
        if not self._log_path or not self._log_rows:
            return
        keys = list(self._log_rows[0].keys())
        with self._log_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._log_rows)
        self.get_logger().info(f"Wrote {len(self._log_rows)} scan log rows to {self._log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",
                   choices=["dropout", "range_clip", "sector_remove", "sector_thin", "yaw_rotate", "translate_y"],
                   default="dropout")
    p.add_argument("--input-topic", default="/sensing/lidar/top/pointcloud_raw_ex")
    p.add_argument("--output-topic", default="/sensing/lidar/top/pointcloud_raw_ex_dropout")
    p.add_argument("--schedule", default="0:0.0,20:0.2,40:0.4,60:0.6,80:0.8",
                   help=(
                       "comma list 't_sec:value' (step-wise constant). "
                       "dropout value=drop ratio [0,1], range_clip value=max radius [m], "
                       "sector_remove value=removed sector width [deg], "
                       "yaw_rotate value=rotation angle [deg, -45..45], "
                       "translate_y value=lateral offset [m, -3..3]."
                   ))
    p.add_argument("--sector-direction", choices=["front", "left", "right"], default="left",
                   help="sector_remove direction in LiDAR frame: front=0 deg, left=90 deg, right=-90 deg")
    p.add_argument("--sector-center-deg", type=float, default=None,
                   help="optional explicit sector center angle [deg], overrides --sector-direction")
    p.add_argument("--sector-thin-width", type=float, default=160.0,
                   help="sector_thin: full angular width [deg] of the thinned sector "
                        "(centered on --sector-direction); schedule value = removal ratio [0,1]")
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--start-delay-sec", type=float, default=0.0,
                   help="extra delay added on top of trigger (or wall-time start if no trigger)")
    p.add_argument("--start-trigger-topic", type=str, default="/control/command/control_cmd",
                   help="topic whose FIRST message resets t=0 (set empty to disable)")
    p.add_argument("--log-csv", type=str, default=None,
                   help="per-scan log csv path")
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args()
    if args.start_trigger_topic in ("", "none", "None"):
        args.start_trigger_topic = None
    if args.sector_center_deg is None:
        args.sector_center_deg = sector_center_deg(args.sector_direction)

    rclpy.init()
    node = LidarDropoutNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.flush_log()
        except Exception as exc:
            print(f"flush_log failed: {exc}", file=sys.stderr)
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
