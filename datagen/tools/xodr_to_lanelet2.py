#!/usr/bin/env python3
"""CARLA OpenDRIVE (xodr) -> Autoware lanelet2_map.osm (Local projector; right-handed frame, y negated).

Writes the schema Autoware expects: node (lat/lon = local/111319.49 plus local_x/local_y/ele tags),
way (boundary polylines), relation (left/right, type=lanelet, subtype=road). No CARLA server is
needed; carla.Map is built from the xodr string.
Usage: xodr_to_lanelet2.py <town.xodr> <lanelet2_map.osm> <name>
"""
import sys
import carla

M2DEG = 1.0 / 111319.49
xodr_path, out_path, name = sys.argv[1], sys.argv[2], sys.argv[3]
cmap = carla.Map(name, open(xodr_path).read())

nodes, ways, rels = [], [], []
nid, wid = [1000], [500000]


def add_node(x, y, z):
    i = nid[0]; nid[0] += 1
    nodes.append((i, x, y, z))
    return i


def add_way(refs):
    i = wid[0]; wid[0] += 1
    ways.append((i, refs))
    return i


rid = [900000]
for wp0, _ in cmap.get_topology():
    seg = [wp0] + wp0.next_until_lane_end(2.0)
    if len(seg) < 2:
        continue
    L, R = [], []
    for w in seg:
        t = w.transform
        x, y = t.location.x, -t.location.y          # left-handed -> right-handed
        z = t.location.z
        yaw = -t.rotation.yaw * 3.141592653589793 / 180.0
        import math
        nxl, nyl = -math.sin(yaw), math.cos(yaw)     # left normal
        hw = w.lane_width / 2.0
        L.append(add_node(x + nxl * hw, y + nyl * hw, z))
        R.append(add_node(x - nxl * hw, y - nyl * hw, z))
    wl, wr = add_way(L), add_way(R)
    i = rid[0]; rid[0] += 1
    rels.append((i, wl, wr))

with open(out_path, "w") as f:
    f.write('<?xml version="1.0"?>\n<osm version="0.6" generator="lanelet2">\n')
    for i, x, y, z in nodes:
        f.write(f'  <node id="{i}" visible="true" version="1" '
                f'lat="{y*M2DEG:.10f}" lon="{x*M2DEG:.10f}">\n'
                f'    <tag k="ele" v="{z:.6f}" />\n'
                f'    <tag k="local_x" v="{x:.6f}" />\n'
                f'    <tag k="local_y" v="{y:.6f}" />\n  </node>\n')
    for i, refs in ways:
        f.write(f'  <way id="{i}" visible="true" version="1">\n')
        for r in refs:
            f.write(f'    <nd ref="{r}" />\n')
        f.write('  </way>\n')
    for i, wl, wr in rels:
        f.write(f'  <relation id="{i}" visible="true" version="1">\n'
                f'    <member type="way" ref="{wl}" role="left" />\n'
                f'    <member type="way" ref="{wr}" role="right" />\n'
                f'    <tag k="subtype" v="road" />\n'
                f'    <tag k="type" v="lanelet" />\n  </relation>\n')
    f.write('</osm>\n')
print(f"lanelets={len(rels)} nodes={len(nodes)} -> {out_path}")
