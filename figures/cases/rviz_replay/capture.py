#!/usr/bin/env python3
"""Render one replay instant in RViz and capture the render window to PNG.

Usage: capture.py RUN T_SEC OUT.png [--scale 90] [--size 1000]
scale = pixels per metre (TopDownOrtho); the view is centred on the GT pose
with the GT heading pointing up.
"""
import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replay_node import load_run  # noqa: E402

RVIZ_TMPL = """Panels: []
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/PointCloud2
      Enabled: true
      Name: Map
      Topic:
        Depth: 1
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /replay/map
      Style: Flat Squares
      Size (m): 0.06
      Color Transformer: FlatColor
      Color: 200; 200; 200
      Use Fixed Frame: true
      Alpha: 1
      Decay Time: 0
      Position Transformer: XYZ
      Queue Size: 1
      Selectable: false
      Value: true
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: Markers
      Namespaces: {{}}
      Topic:
        Depth: 1
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /replay/markers
      Value: true
  Enabled: true
  Global Options:
    Background Color: 255; 255; 255
    Fixed Frame: map
    Frame Rate: 30
  Name: root
  Tools: []
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/TopDownOrtho
      Angle: {angle}
      Scale: {scale}
      X: {x}
      Y: {y}
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.06
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.01
      Target Frame: <Fixed Frame>
      Value: TopDownOrtho (rviz_default_plugins)
    Saved: ~
Window Geometry:
  Displays:
    collapsed: true
  Height: {size}
  Hide Left Dock: true
  Hide Right Dock: true
  Width: {size}
  X: 100
  Y: 100
  collapsed: false
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('t', type=float); ap.add_argument('out')
    ap.add_argument('--scale', type=float, default=90.0)
    ap.add_argument('--size', type=int, default=1000)
    ap.add_argument('--angle_offset', type=float, default=0.0)
    ap.add_argument('--route', default='routeA')
    a = ap.parse_args()
    d = load_run(a.run, a.route)
    import numpy as np
    i = int(np.argmin(np.abs(d['t'] - a.t)))
    yaw = math.radians(d['gt_yaw'][i])
    # TopDownOrtho: Angle rotates the camera about Z; heading-up => angle = yaw - pi/2 (+offset)
    angle = yaw - math.pi / 2 + a.angle_offset
    cfg = HERE / '_view.rviz'
    cfg.write_text(RVIZ_TMPL.format(angle=angle, scale=a.scale, x=d['gt_x'][i], y=d['gt_y'][i],
                                    size=a.size))
    env = dict(os.environ)
    node = subprocess.Popen([sys.executable, str(HERE / 'replay_node.py'), a.run, str(a.t), '--route', a.route], env=env)
    rviz = subprocess.Popen(['rviz2', '-d', str(cfg)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        geom = None
        for _ in range(40):
            time.sleep(0.5)
            r = subprocess.run(['xwininfo', '-root', '-tree'], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if '_view.rviz' in line or 'RViz' in line:
                    m = re.search(r'(\d+)x(\d+)\+(-?\d+)\+(-?\d+)\s+\+(-?\d+)\+(-?\d+)', line)
                    if m and int(m.group(1)) > 300:
                        geom = tuple(int(v) for v in m.groups())
            if geom:
                break
        if not geom:
            sys.exit('rviz window not found')
        time.sleep(6)                       # let the cloud/mesh load and render
        w, h, _, _, ax, ay = geom
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'x11grab',
                        '-video_size', f'{w}x{h}', '-i', f':0.0+{ax},{ay}',
                        '-frames:v', '1', a.out], check=True)
        print('captured', a.out, geom)
    finally:
        rviz.terminate(); node.terminate()
        rviz.wait(timeout=10); node.wait(timeout=10)


if __name__ == '__main__':
    main()
