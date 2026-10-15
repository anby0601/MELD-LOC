#!/usr/bin/env python3
"""Mapping sweep: build pointcloud_map.pcd of a CARLA town by accumulating LiDAR scans at ground-truth poses.

The point clouds shipped with CARLA do not register well enough with the run-time sensor
(NVTL below the matcher threshold), so the maps are built with the run-time LiDAR specification.

Method: teleport the ego vehicle along every lane waypoint (2 m spacing by default), take one
revolution with the same LiDAR as the run-time bridge (64 ch, 300k pps, +10..-30 deg, 20 Hz, no
noise, mounted at x=-2.425 z=1.0), transform to world, negate y (right-handed), remove duplicates
with a 0.1 m voxel, drop ego reflections within 3 m, write a binary PCD.
"""
import argparse
import math
import queue
import sys
import time

import numpy as np

import carla

Q = 0.1          # voxel (m)
SELF_R = 3.0     # ego-reflection removal radius (sensor frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--voxel", type=float, default=Q)
    a = ap.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    if a.town not in world.get_map().name:
        world = client.load_world(a.town)
        time.sleep(3.0)
    st = world.get_settings()
    st.synchronous_mode = True
    st.fixed_delta_seconds = 0.05          # 20 Hz rotation x 0.05 s = one revolution per tick
    st.no_rendering_mode = True
    world.apply_settings(st)

    bp = world.get_blueprint_library().filter("vehicle.*model3*")[0]
    cmap = world.get_map()
    wps = cmap.generate_waypoints(a.spacing)
    print(f"waypoints={len(wps)} spacing={a.spacing}", flush=True)
    t0 = wps[0].transform
    ego = world.spawn_actor(bp, carla.Transform(
        carla.Location(t0.location.x, t0.location.y, t0.location.z + 0.5),
        t0.rotation))
    ego.set_simulate_physics(False)

    lbp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    for k, v in {"range": "100", "channels": "64", "points_per_second": "300000",
                 "upper_fov": "10.0", "lower_fov": "-30.0",
                 "rotation_frequency": "20", "noise_stddev": "0.0"}.items():
        lbp.set_attribute(k, v)
    lidar = world.spawn_actor(
        lbp, carla.Transform(carla.Location(x=-2.425, y=0.0, z=1.0)), attach_to=ego)
    q = queue.Queue()
    lidar.listen(q.put)

    vox = a.voxel
    key_chunks = []
    pending = []
    n_scan = 0

    def flush():
        nonlocal key_chunks, pending
        if pending:
            key_chunks.append(np.concatenate(pending))
            pending = []
        if len(key_chunks) > 1:
            key_chunks = [np.unique(np.concatenate(key_chunks))]

    t_start = time.time()
    for i, wp in enumerate(wps):
        tf = wp.transform
        ego.set_transform(carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z + 0.5),
            tf.rotation))
        world.tick()
        while not q.empty():
            q.get_nowait()                 # discard the frame captured before the teleport
        world.tick()
        try:
            d = q.get(timeout=5.0)
        except queue.Empty:
            continue
        pts = np.frombuffer(d.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        keep = np.linalg.norm(pts, axis=1) > SELF_R
        pts = pts[keep]
        if not len(pts):
            continue
        M = np.array(d.transform.get_matrix())
        pw = pts @ M[:3, :3].T + M[:3, 3]
        x, y, z = pw[:, 0], -pw[:, 1], pw[:, 2]   # left-handed -> right-handed
        ix = np.floor(x / vox).astype(np.int64) + (1 << 20)
        iy = np.floor(y / vox).astype(np.int64) + (1 << 20)
        iz = np.floor(z / vox).astype(np.int64) + (1 << 20)
        pending.append((ix << 42) | (iy << 21) | iz)
        n_scan += 1
        if len(pending) >= 100:
            flush()
        if i % 500 == 0:
            nk = sum(len(c) for c in key_chunks) + sum(len(p) for p in pending)
            print(f"[{time.time()-t_start:6.1f}s] wp {i}/{len(wps)} scans={n_scan} "
                  f"~keys={nk}", flush=True)

    flush()
    keys = key_chunks[0]
    iz = (keys & ((1 << 21) - 1)) - (1 << 20)
    iy = ((keys >> 21) & ((1 << 21) - 1)) - (1 << 20)
    ix = (keys >> 42) - (1 << 20)
    pts = np.stack([(ix + 0.5) * vox, (iy + 0.5) * vox, (iz + 0.5) * vox],
                   axis=1).astype(np.float32)
    hdr = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
           "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
           f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
           f"POINTS {len(pts)}\nDATA binary\n")
    with open(a.out, "wb") as f:
        f.write(hdr.encode())
        f.write(pts.tobytes())
    print(f"DONE scans={n_scan} points={len(pts)} -> {a.out}", flush=True)

    lidar.destroy()
    ego.destroy()
    st.synchronous_mode = False
    st.no_rendering_mode = False
    world.apply_settings(st)


if __name__ == "__main__":
    main()
