#!/usr/bin/env python3
"""Speed-profile variants of routeA (evaluation-only stress sets).

std   : the original profile (v_ref 1.2-5.2 m/s, mean 4.39 m/s), unchanged
fast  : v_ref x 1.35, capped at 7.0 m/s (start/stop ramps kept): higher cruise speed
aggr  : alternates 2.0 and 5.5 m/s every 60 m of progress: repeated hard acceleration and braking (last 60 m = original stop ramp)

The route geometry (waypoints x/y/yaw/s) is unchanged, so the progress-based perturbation schedules fire at the same places.
"""
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "routes" / "routeA.json"
OUT = Path(__file__).resolve().parents[1] / "routes"


def main():
    d = json.loads(SRC.read_text())
    wp = d["waypoints"]
    L = wp[-1]["s"]
    OUT.mkdir(parents=True, exist_ok=True)
    # fast
    f = json.loads(SRC.read_text())
    for w in f["waypoints"]:
        v = w["v_ref_mps"]
        if w["s"] < 70.0 or w["s"] > L - 45.0:        # keep the original start/stop ramps
            continue
        w["v_ref_mps"] = round(min(v * 1.35, 7.0), 3)
    f["speed_profile"] = dict(d["speed_profile"], variant="fast_x1.35_cap7.0", max_profile_speed_mps=7.0)
    f["name"] = d["name"] + "_fast"
    (OUT / "routeA_fast.json").write_text(json.dumps(f, indent=1))
    # aggressive
    a = json.loads(SRC.read_text())
    for w in a["waypoints"]:
        s = w["s"]
        if s < 20.0 or s > L - 60.0:
            continue
        seg = int((s - 20.0) // 60.0)
        w["v_ref_mps"] = 5.5 if seg % 2 == 0 else 2.0
    a["speed_profile"] = dict(d["speed_profile"], variant="aggr_alt_5.5_2.0_every60m")
    a["name"] = d["name"] + "_aggr"
    (OUT / "routeA_aggr.json").write_text(json.dumps(a, indent=1))
    for name, r in (("std", d), ("fast", f), ("aggr", a)):
        v = [w["v_ref_mps"] for w in r["waypoints"]]
        print(f"{name}: n={len(v)} v_ref min {min(v):.2f} max {max(v):.2f} mean {sum(v)/len(v):.2f}  L={L:.1f} m")


if __name__ == "__main__":
    main()
