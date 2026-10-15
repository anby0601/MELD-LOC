#!/usr/bin/env python3
"""Replay one logged instant into RViz (frozen logs + frozen model outputs).

Publishes (latched):
  /replay/map      PointCloud2  point-cloud map crop around the GT pose
  /replay/markers  MarkerArray  GT vehicle mesh, NDT-estimated vehicle mesh,
                                fixed 3-sigma box, MELD-Loc q90 box (both on the estimate)
Usage: replay_node.py RUN T_SEC [--radius 40]
"""
import argparse
import csv
import math
import struct
import os
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

HERE = Path(__file__).resolve().parents[2]
CACHE = HERE / 'figures/out/_hero_cache'
MESH = ('file://' + os.path.join(os.environ.get('AUTOWARE_WS', os.path.expanduser('~/autoware')), 'install/autoware_sample_vehicle_description')
        + ''
        '/share/autoware_sample_vehicle_description/mesh/lexus.dae')
ROUTES = {   # frozen route constants (LON_OFFSETS in build_features.py)
    'routeA': dict(logdir='holdout', off=-1.6512603049722019,
                   q='holdout_q90.npz', map='town01_crop.npz'),
    'routeB': dict(logdir='crossroute', off=-1.5855934394574358,
                   q='crossroute_q90.npz', map='town02_crop.npz'),
}
BL = 0.45
WHEEL_BASE, FRONT_OH, REAR_OH, WIDTH = 2.79, 1.0, 1.1, 1.896


def load_run(run, route='routeA'):
    R = ROUTES[route]
    cols = ('timestamp', 'gt_x', 'gt_y', 'gt_yaw', 'ndt_x', 'ndt_y', 'ndt_yaw')
    rows = []
    with open(HERE / 'data/logs' / R['logdir'] / f'{run}_a1/long_route_pilot.csv') as fh:
        for r in csv.DictReader(fh):
            if r['phase'] != 'route':
                continue
            try:
                rows.append([float(r[c]) for c in cols])
            except ValueError:
                continue
    a = np.array(rows)
    a = a[np.all(np.isfinite(a), axis=1)]
    d = dict(zip(cols, a.T))
    th = np.radians(d['gt_yaw'])
    d['cx'] = d['ndt_x'] - R['off'] * np.cos(th)
    d['cy'] = d['ndt_y'] - R['off'] * np.sin(th)
    d['t'] = d['timestamp'] - d['timestamp'][0]
    z = np.load(CACHE / R['q'], allow_pickle=True)
    m = z['scenario'] == run
    o = np.argsort(z['t'][m])
    for ax in ('lon', 'lat'):
        d[f'q_{ax}'] = np.interp(d['timestamp'], z['t'][m][o], z[f'q90_{ax}'][m][o])
    return d


def quat_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return (w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2)


def quat_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r/2), math.sin(r/2), math.cos(p/2),
                              math.sin(p/2), math.cos(y/2), math.sin(y/2))
    return (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)


class Replay(Node):
    def __init__(self, run, tsec, radius, route):
        super().__init__('replay')
        d = load_run(run, route)
        i = int(np.argmin(np.abs(d['t'] - tsec)))
        self.gt = (d['gt_x'][i], d['gt_y'][i], math.radians(d['gt_yaw'][i]))
        self.est = (d['cx'][i], d['cy'][i], math.radians(d['ndt_yaw'][i]))
        self.q = (d['q_lon'][i], d['q_lat'][i])
        print(f'GT {self.gt}  EST {self.est}  q90 lon/lat {self.q}', flush=True)
        mp = np.load(CACHE / ROUTES[route]['map'])
        m = (np.abs(mp['x'] - self.gt[0]) < radius) & (np.abs(mp['y'] - self.gt[1]) < radius)
        self.pts = np.c_[mp['x'][m], mp['y'][m], mp['z'][m]].astype(np.float32)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_map = self.create_publisher(PointCloud2, '/replay/map', qos)
        self.pub_mk = self.create_publisher(MarkerArray, '/replay/markers', qos)
        self.timer = self.create_timer(1.0, self.tick)

    def cloud(self):
        h = Header(frame_id='map', stamp=self.get_clock().now().to_msg())
        msg = PointCloud2(header=h, height=1, width=len(self.pts), is_dense=True,
                          is_bigendian=False, point_step=12, row_step=12*len(self.pts))
        msg.fields = [PointField(name=n, offset=4*k, datatype=PointField.FLOAT32, count=1)
                      for k, n in enumerate('xyz')]
        msg.data = self.pts.tobytes()
        return msg

    def vehicle(self, mid, pose, rgba):
        x, y, yaw = pose
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id, m.type, m.action = 'vehicle', mid, Marker.MESH_RESOURCE, Marker.ADD
        m.mesh_resource = MESH
        m.mesh_use_embedded_materials = False
        # xacro: origin xyz=(wheel_base/2,0,0) rpy=(pi/2,0,pi) relative to base_link
        m.pose.position.x = x + (WHEEL_BASE/2) * math.cos(yaw)
        m.pose.position.y = y + (WHEEL_BASE/2) * math.sin(yaw)
        m.pose.position.z = 0.0
        q = quat_mul(quat_rpy(0, 0, yaw), quat_rpy(math.pi/2, 0, math.pi))
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def box(self, mid, pose, hl, hw, rgba, width, z=0.3):
        """Rectangle around the vehicle footprint (base_link at the rear axle) enlarged by hl/hw."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        xs = (-REAR_OH - hl, WHEEL_BASE + FRONT_OH + hl)
        ys = (-WIDTH/2 - hw, WIDTH/2 + hw)
        corners = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1]), (xs[0], ys[0])]
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id, m.type, m.action = 'bound', mid, Marker.LINE_STRIP, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = width
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        for u, v in corners:
            m.points.append(Point(x=x + c*u - s*v, y=y + s*u + c*v, z=z))
        return m

    def tick(self):
        self.pub_map.publish(self.cloud())
        ma = MarkerArray()
        ma.markers.append(self.vehicle(0, self.gt, (0.25, 0.25, 0.25, 1.0)))
        ma.markers.append(self.vehicle(1, self.est, (0.78, 0.06, 0.18, 0.55)))
        ma.markers.append(self.box(2, self.est, self.q[0], self.q[1], (0.91, 0.49, 0.05, 0.95), 0.18))
        ma.markers.append(self.box(3, self.est, BL, BL, (0.12, 0.37, 0.75, 1.0), 0.12))
        self.pub_mk.publish(ma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('t', type=float)
    ap.add_argument('--radius', type=float, default=40.0)
    ap.add_argument('--route', default='routeA')
    a = ap.parse_args()
    rclpy.init()
    n = Replay(a.run, a.t, a.radius, a.route)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
