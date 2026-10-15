#!/usr/bin/env python3
"""NON-RIGID LiDAR distortion kernels — the third perturbation category.

No host needed: --selftest of this file runs with numpy only.

Why this category exists
------------------------
The cohort so far has exactly two categories:

  subtractive  range_clip / dropout / sector_remove / sector_thin
               geometry is REMOVED -> the fit degrades -> transform_probability and iteration_num
               respond (measured: 9-204 alarms/h).  These move the LONGITUDINAL axis.

  rigid        translate_y / yaw_rotate
               the cloud is moved by a single rigid transform.  NDT is a rigid registration, so the
               transformed cloud fits the map EXACTLY as well at the transformed pose: the score is
               unchanged (measured: tp/iteration at control level) while the pose is biased.  These
               move the LATERAL and YAW axes and are invisible to every internal indicator.

That is why lateral/yaw risk has no observable cause in the current data.  A NON-RIGID distortion
breaks the rigid-registration assumption: no single rigid transform can align a sheared cloud, so
    (a) the residual — and therefore transform_probability / iteration_num — must degrade, and
    (b) the best-fit pose is biased in the sheared direction.
Both at once. That is exactly the missing regime.

Physical motivation (these are real failure modes, not arbitrary maths):
  * motion-distortion (de-skew) failure while turning
  * LiDAR-to-vehicle time-sync error during rotation
  * intra-scan extrinsic error

Kernels
-------
  yaw_shear       theta(r) = shear_deg_at_ref * (r / ref_range_m), rotate (x,y) about z by theta(r).
                  Range r is invariant under a z-rotation, so a `distance` field stays consistent.
  lateral_shear   y' = y + shear_m_at_ref * (r / ref_range_m).
                  NOTE: this changes r, so a `distance` field would become stale — the same caveat
                  the frozen `apply_lateral_translation` already carries. Only y is rewritten.

Severity is quoted AT A REFERENCE RANGE so the number is comparable with the rigid families
(`yaw_shear 2.0` means "a point at ref_range is rotated 2 deg"; near points are barely moved).
"""
from __future__ import annotations

import argparse
import math

import numpy as np

DEFAULT_REF_RANGE_M = 50.0

# validation ranges, mirroring the rigid families they are meant to be compared against
LIMITS = {
    "yaw_shear": (-45.0, 45.0),        # deg at the reference range, cf. yaw_rotate
    "lateral_shear": (-3.0, 3.0),      # m   at the reference range, cf. translate_y
    "longitudinal_shear": (-3.0, 3.0),  # m at the reference range, lon analogue of lateral_shear
    "translate_x": (-8.0, 8.0),        # m rigid forward offset, lon analogue of translate_y
}


# ----------------------------------------------------------------- pure maths (ROS-free, testable)
def shear_xy(x, y, shear_deg_at_ref, ref_range_m=DEFAULT_REF_RANGE_M):
    """range-proportional rotation about z. Returns (x', y'). r is preserved exactly."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.hypot(x, y)
    th = math.radians(float(shear_deg_at_ref)) * (r / float(ref_range_m))
    c, s = np.cos(th), np.sin(th)
    return x * c - y * s, x * s + y * c


def shear_y(x, y, shear_m_at_ref, ref_range_m=DEFAULT_REF_RANGE_M):
    """range-proportional lateral offset. Returns y'. r is NOT preserved."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.hypot(x, y)
    return y + float(shear_m_at_ref) * (r / float(ref_range_m))


def best_fit_rigid_residual(x0, y0, x1, y1):
    """RMS residual after the OPTIMAL 2-D rigid alignment of (x0,y0) onto (x1,y1).

    This is the quantity NDT cannot drive to zero.  For a rigid perturbation it is 0 by
    construction; for a shear it is strictly positive and grows with the shear.  Kabsch in 2-D.
    """
    p = np.stack([np.asarray(x0, float), np.asarray(y0, float)])
    q = np.stack([np.asarray(x1, float), np.asarray(y1, float)])
    pc, qc = p.mean(1, keepdims=True), q.mean(1, keepdims=True)
    p0, q0 = p - pc, q - qc
    h = p0 @ q0.T
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, d]) @ u.T
    resid = q0 - rot @ p0
    return float(np.sqrt((resid ** 2).sum(0).mean())), rot


def validate(mode, value):
    lo, hi = LIMITS[mode]
    if not (lo <= value <= hi):
        raise ValueError(f"{mode} value must be in [{lo}, {hi}]: {value}")
    return value


# ----------------------------------------------------------------- PointCloud2 wrappers
def _views(msg, raw_points, names):
    """writable byte buffer + strided views for the named float fields"""
    import sys
    sr = str(__import__("pathlib").Path(__file__).resolve().parent / "awsim_kernels")
    if sr not in sys.path:
        sys.path.insert(0, sr)
    import lidar_dropout_node as awsim                                   # frozen kernels, reused

    buf = bytearray(raw_points.tobytes())
    step = int(msg.point_step)
    n = raw_points.shape[0]
    out = {}
    for nm in names:
        f = next(fl for fl in msg.fields if fl.name == nm)
        dt = awsim._field_dtype(f, msg.is_bigendian)
        out[nm] = np.ndarray(shape=(n,), dtype=dt, buffer=buf, offset=int(f.offset), strides=(step,))
    return buf, out, awsim


def apply_yaw_shear(msg, raw_points, shear_deg_at_ref, ref_range_m=DEFAULT_REF_RANGE_M):
    """Range-proportional yaw shear. Keeps all points; rewrites only x and y."""
    n = raw_points.shape[0]
    if n == 0 or shear_deg_at_ref == 0.0:
        return raw_points.tobytes()
    buf, v, awsim = _views(msg, raw_points, ("x", "y"))
    x = awsim._get_numeric_field(msg, "x", raw_points)
    y = awsim._get_numeric_field(msg, "y", raw_points)
    if x is None or y is None:
        raise ValueError("yaw_shear requires x/y fields")
    xr, yr = shear_xy(x, y, shear_deg_at_ref, ref_range_m)
    v["x"][:] = xr.astype(v["x"].dtype)
    v["y"][:] = yr.astype(v["y"].dtype)
    return bytes(buf)


def apply_lateral_shear(msg, raw_points, shear_m_at_ref, ref_range_m=DEFAULT_REF_RANGE_M):
    """Range-proportional lateral shear. Keeps all points; rewrites only y."""
    n = raw_points.shape[0]
    if n == 0 or shear_m_at_ref == 0.0:
        return raw_points.tobytes()
    buf, v, awsim = _views(msg, raw_points, ("x", "y"))
    x = awsim._get_numeric_field(msg, "x", raw_points)
    y = awsim._get_numeric_field(msg, "y", raw_points)
    if x is None or y is None:
        raise ValueError("lateral_shear requires x/y fields")
    v["y"][:] = shear_y(x, y, shear_m_at_ref, ref_range_m).astype(v["y"].dtype)
    return bytes(buf)


# ----------------------------------------------------------------- self-test (no ROS needed)
def selftest():
    rng = np.random.default_rng(20260801)
    # a synthetic scan: 32 rings x 900 azimuths, ranges 3..60 m, like a spinning LiDAR
    az = np.deg2rad(np.linspace(-180, 180, 900, endpoint=False))
    r = rng.uniform(3.0, 60.0, size=(32, az.size))
    x = (r * np.cos(az)).ravel()
    y = (r * np.sin(az)).ravel()
    ok = True

    print("=== 1. rigid controls: the best-fit rigid residual MUST be ~0 ===")
    for deg in (0.5, 2.0, 5.0):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        res, _ = best_fit_rigid_residual(x, y, x * c - y * s, x * s + y * c)
        print(f"   yaw_rotate {deg:>4} deg   best-fit rigid residual {res:.3e} m")
        ok &= res < 1e-9
    for m in (0.2, 1.0):
        res, _ = best_fit_rigid_residual(x, y, x, y + m)
        print(f"   translate_y {m:>4} m     best-fit rigid residual {res:.3e} m")
        ok &= res < 1e-9

    print("\n=== 2. yaw_shear: residual must be POSITIVE and grow with severity ===")
    print(f"   {'shear@50m':>11}{'rigid resid':>14}{'best-fit yaw':>14}{'max disp':>11}{'disp@10m':>11}")
    prev = -1.0
    for deg in (0.25, 0.5, 1.0, 2.0, 3.0):
        xs, ys = shear_xy(x, y, deg)
        res, rot = best_fit_rigid_residual(x, y, xs, ys)
        fit_yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
        disp = np.hypot(xs - x, ys - y)
        near = np.hypot(x, y) < 11.0
        print(f"   {deg:>11.2f}{res:>14.4f}{fit_yaw:>13.3f}d{disp.max():>11.3f}{disp[near].mean():>11.3f}")
        ok &= res > prev
        prev = res

    print("\n=== 3. range is preserved by yaw_shear (a `distance` field stays valid) ===")
    xs, ys = shear_xy(x, y, 3.0)
    dr = np.abs(np.hypot(xs, ys) - np.hypot(x, y))
    print(f"   max |dr| = {dr.max():.3e} m")
    ok &= dr.max() < 1e-9

    print("\n=== 4. lateral_shear ===")
    print(f"   {'shear@50m':>11}{'rigid resid':>14}{'best-fit yaw':>14}")
    prev = -1.0
    for m in (0.1, 0.25, 0.5, 1.0):
        ys2 = shear_y(x, y, m)
        res, rot = best_fit_rigid_residual(x, y, x, ys2)
        print(f"   {m:>11.2f}{res:>14.4f}{math.degrees(math.atan2(rot[1,0], rot[0,0])):>13.3f}d")
        ok &= res > prev
        prev = res

    print("\n=== 5. limits ===")
    for mode, v, should in (("yaw_shear", 3.0, True), ("yaw_shear", 90.0, False),
                            ("lateral_shear", 0.5, True), ("lateral_shear", 9.0, False)):
        try:
            validate(mode, v); got = True
        except ValueError:
            got = False
        print(f"   validate({mode}, {v}) -> {got}   expected {should}")
        ok &= got == should

    print(f"\nSELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    raise SystemExit(selftest() if a.selftest else 0)


# ------------------------------------------------------------------ longitudinal family
def shear_x(x, y, shear_m_at_ref, ref_range_m=DEFAULT_REF_RANGE_M):
    """range-proportional LONGITUDINAL offset. Returns x'. r is NOT preserved."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.hypot(x, y)
    return x + float(shear_m_at_ref) * (r / float(ref_range_m))


def apply_longitudinal_shear(msg, raw_points, shear_m_at_ref, ref_range_m=DEFAULT_REF_RANGE_M):
    """Range-proportional longitudinal shear. Keeps all points; rewrites only x."""
    n = raw_points.shape[0]
    if n == 0 or shear_m_at_ref == 0.0:
        return raw_points.tobytes()
    buf, v, awsim = _views(msg, raw_points, ("x", "y"))
    x = awsim._get_numeric_field(msg, "x", raw_points)
    y = awsim._get_numeric_field(msg, "y", raw_points)
    if x is None or y is None:
        raise ValueError("longitudinal_shear requires x/y fields")
    v["x"][:] = shear_x(x, y, shear_m_at_ref, ref_range_m).astype(v["x"].dtype)
    return bytes(buf)


def apply_longitudinal_translation(msg, raw_points, offset_m):
    """Rigid +x (forward) offset — lon analogue of translate_y. Rewrites only x."""
    n = raw_points.shape[0]
    if n == 0 or offset_m == 0.0:
        return raw_points.tobytes()
    buf, v, awsim = _views(msg, raw_points, ("x", "y"))
    x = awsim._get_numeric_field(msg, "x", raw_points)
    if x is None:
        raise ValueError("translate_x requires x field")
    v["x"][:] = (np.asarray(x, dtype=np.float64) + float(offset_m)).astype(v["x"].dtype)
    return bytes(buf)
