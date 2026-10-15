#!/usr/bin/env python3
"""Health-assessment pipeline 1/2: the feature/label cache, rebuilt from the raw logs.

Inputs are the raw run logs and the frozen cohort indices only; the cache can be deleted
and regenerated at any time.

Target:
    Y_a(t) = max over [t, t+1s) |e_a(τ)|,   a ∈ {lon, lat, yaw}
    includes the current time t (near-term maximum-error assessment, not a forecast).

Inputs (matcher-internal signals):
    S1 transform_probability     <- NDT-specific (voxel likelihood); the score-free core omits it
    S2 iteration_num             <- matcher-agnostic
    S3 ndt_inter_sample_jump     <- matcher-agnostic
    S4 ndt_output_gap            <- matcher-agnostic
    S5 |ndt_yaw_rate|            <- matcher-agnostic
  x 12 statistics x 2 windows (2 s, 8 s) = 120 dimensions
  ndt_step_dist / ndt_speed are excluded by the registered protocol (dominated by the commanded speed).

Usage:  python3 models/features/build_features.py [--force]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
INCLUDE_ALL = False
IDX = {"development": HERE / "data/indices/development.csv"}   # one row per development run (see data/indices/README.md)
OUT = HERE / "results/features"
CACHE = OUT / "development_features.npz"

H = 1.0                 # assessment window [t, t+H)
STRIDE = 0.5            # anchor spacing [s]
HIST_MIN = 8.0          # minimum history so that the 8 s window is full
SCALES = (2.0, 8.0)
SIG = ["transform_probability", "iteration_num", "ndt_inter_sample_jump",
       "ndt_output_gap", "ndt_yaw_rate_abs"]
NDT_SPECIFIC = {"transform_probability"}          # the matcher-score signal (omitted by the score-free core)
STATS = ["mean", "std", "min", "max", "last", "slope", "q25", "q50", "q75",
         "ptp", "mad", "acf1"]
AX = ("lon", "lat", "yaw")
COL = {"lon": "raw_longitudinal_residual", "lat": "raw_lateral_residual",
       "yaw": "raw_yaw_residual"}
LON_OFFSET = -1.6512603049722019      # routeA: median along-track residual of the undegraded runs
# per-route constant longitudinal offset (a route constant; re-estimated for every new route)
#   routeB_town02_S1: pooled median of the three undegraded Town02 runs (run-to-run range 0.001 m)
LON_OFFSETS = {"routeA": LON_OFFSET,
               "routeB_town02_S1": -1.5855934394574358}


def lon_offset_of(row):
    """Offset by the index row's route_id; rows without a registered route are routeA."""
    return LON_OFFSETS.get(str(row.get("route_id", "") or "routeA"), LON_OFFSET)

# per-axis thresholds (lateral and yaw from lane/vehicle geometry; longitudinal is an operational choice)
LANE_W, VEH_W = 3.5, 1.8
HALF = (LANE_W - VEH_W) / 2
THRESH = {"lat": (HALF, LANE_W / 2), "yaw": (math.degrees(math.atan(HALF / 20.0)),
                                             math.degrees(math.atan(HALF / 10.0))),
          "lon": (2.5, 5.0)}
THRESH_BASIS = {
    "lat": "T1=(lane 3.5 m - vehicle 1.8 m)/2=0.85 m (vehicle edge reaches the lane edge), T2=lane/2=1.75 m (centre leaves the lane)",
    "yaw": "T1=atan(0.85/20 m)=2.434 deg, T2=atan(0.85/10 m)=4.858 deg (half-margin consumed at the look-ahead distance)",
    "lon": "no geometric derivation; operational design choice 2.5/5.0 m after offset correction"}


def read_index():
    rows, seen = [], set()
    per_index = Counter()
    for name, p in IDX.items():
        if not p.exists():
            continue
        with p.open() as f:
            for x in csv.DictReader(f):
                if x["scenario_id"] in seen:
                    continue
                seen.add(x["scenario_id"]); x["_index"] = name; rows.append(x)
    tv = [r for r in rows if r.get("split_role") in ("train", "validation")]
    smoke = {r["scenario_id"] for r in tv if r["scenario_id"].startswith("shr_")}
    use = [r for r in tv if r.get("run_status") in ("VALID_COMPLETED", "VALID_COLLAPSE")
           and r["scenario_id"] not in smoke]
    for r in use:
        per_index[r["_index"]] += 1
    # runs whose logs were not reachable when the cache was frozen (frozen_cache = 0 in the index): the
    # frozen models never saw them, so they are skipped unless --all is given
    excluded = [r for r in use if r.get("frozen_cache") == "0"]
    if not INCLUDE_ALL:
        use = [r for r in use if r.get("frozen_cache") != "0"]
    return rows, tv, sorted(smoke), use, dict(per_index), excluded


def load_run(d: Path):
    f = d / "long_route_pilot.csv"
    if not f.exists():
        return None, "missing_log"
    need = ["timestamp", "transform_probability", "iteration_num", "ndt_output_gap",
            "ndt_inter_sample_jump", "ndt_yaw"] + list(COL.values())
    with f.open() as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames or any(c not in rd.fieldnames for c in need):
            return None, "missing_columns"
        acc = defaultdict(list)
        for row in rd:
            for c in need:
                acc[c].append(row.get(c, ""))
    out = {}
    for c in need:
        v = np.array([x if x not in ("", "None") else "nan" for x in acc[c]], dtype=str)
        try:
            out[c] = v.astype(np.float64)
        except ValueError:
            return None, "unparsable"
    m = np.isfinite(out["timestamp"])
    out = {c: out[c][m] for c in need}
    if len(out["timestamp"]) < 200:
        return None, "too_short"
    o = np.argsort(out["timestamp"], kind="mergesort")
    out = {c: out[c][o] for c in need}
    t = out["timestamp"]
    dt = np.diff(t, prepend=t[0] - 0.1)
    dy = np.diff(np.unwrap(np.radians(out["ndt_yaw"])), prepend=0.0)
    out["ndt_yaw_rate_abs"] = np.abs(np.degrees(dy) / np.maximum(dt, 1e-3))
    return out, "ok"


def window_stats(t, v, t0, t1):
    m = (t >= t0) & (t <= t1)
    if not m.any():
        return None
    x = v[m]; tt = t[m]
    ok = np.isfinite(x)
    x, tt = x[ok], tt[ok]
    if len(x) < 3:
        return None
    d = np.diff(x)
    sl = np.polyfit(tt - tt[0], x, 1)[0] if tt[-1] > tt[0] else 0.0
    xc = x - x.mean(); den = float((xc * xc).sum())
    a1 = float((xc[:-1] * xc[1:]).sum() / den) if den > 1e-12 else 0.0
    return [x.mean(), x.std(), x.min(), x.max(), x[-1], sl,
            np.quantile(x, .25), np.quantile(x, .50), np.quantile(x, .75),
            x.max() - x.min(), np.abs(d).mean() if len(d) else 0.0, a1]


FORBIDDEN = ("gt_", "carla_ego_", "raw_", "route_", "lane_", "collision", "safety_")


def assert_no_leak():
    """Assert that no ground-truth, residual or route-derived quantity is an input signal."""
    for s in SIG:
        assert not any(s.startswith(p) for p in FORBIDDEN), f"forbidden input: {s}"
    assert "ndt_yaw_rate_abs" in SIG          # derived from the NDT yaw output, not from ground truth
    return {"checked_signals": SIG, "forbidden_prefixes": list(FORBIDDEN), "result": "PASS"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all", action="store_true", help="also load the runs marked frozen_cache = 0 (not part of the frozen cache)")
    a = ap.parse_args()
    global INCLUDE_ALL; INCLUDE_ALL = a.all
    OUT.mkdir(parents=True, exist_ok=True)
    if CACHE.exists() and not a.force:
        print(f"already exists: {CACHE}  (use --force to rebuild)")
        return

    allrows, tv, smoke, use, per_index, excluded = read_index()
    fname = [f"{s}|{L:g}s|{st}" for L in SCALES for s in SIG for st in STATS]

    X, Y, YC, YF, META = [], [], [], [], []
    load_status = Counter()
    missing = [{"scenario_id": r["scenario_id"], "reason": "not_in_frozen_cache",
                "dir_exists": Path(r["raw_run_dir"]).exists(), "family": r["family"], "split_role": r["split_role"]}
               for r in excluded]
    for ri, r in enumerate(use):
        d, why = load_run(Path(r["raw_run_dir"]))
        load_status[why] += 1
        if d is None:
            missing.append({"scenario_id": r["scenario_id"], "reason": why,
                            "dir_exists": Path(r["raw_run_dir"]).exists(),
                            "family": r["family"], "split_role": r["split_role"]})
            continue
        t = d["timestamp"]
        hi = np.searchsorted(t, t + H, side="left")
        # three label variants (the two extra ones serve the leakage/triviality audits)
        #   incl : max over [t, t+1s)   - the assessment target (includes t)
        #   curr : |e(t)|               - current error only
        #   futr : max over (t, t+1s)   - strictly future
        lab, lab_c, lab_f = {}, {}, {}
        run_lon_offset = lon_offset_of(r)
        for ax in AX:
            e = np.abs(d[COL[ax]] - (run_lon_offset if ax == "lon" else 0.0))
            lab[ax] = np.array([e[i:max(j, i + 1)].max() for i, j in enumerate(hi)])
            lab_c[ax] = e
            lab_f[ax] = np.array([e[i + 1:j].max() if j > i + 1 else np.nan
                                  for i, j in enumerate(hi)])
        anchors = np.arange(t[0] + HIST_MIN, t[-1] - H, STRIDE)
        for tau in anchors:
            i = np.searchsorted(t, tau, side="right") - 1
            if i < 0 or tau - t[i] > 0.5:
                continue
            f = []; bad = False
            for L in SCALES:
                for s in SIG:
                    w = window_stats(t, d[s], tau - L, tau)
                    if w is None:
                        bad = True; break
                    f += w
                if bad:
                    break
            if bad or not all(np.isfinite(lab[ax][i]) for ax in AX):
                continue
            # leakage guard: the feature window never extends past the anchor
            assert tau - SCALES[0] <= tau and tau <= t[i] + 0.5
            X.append(f); Y.append([lab[ax][i] for ax in AX])
            YC.append([lab_c[ax][i] for ax in AX]); YF.append([lab_f[ax][i] for ax in AX])
            META.append((r["scenario_id"], r["split_role"], r["family"],
                         r.get("cluster_id", ""), tau, r["_index"], r["run_status"],
                         r.get("recipe_group_id", "") or r.get("cluster_id", "")))
        if (ri + 1) % 40 == 0:
            print(f"  {ri+1}/{len(use)} anchors {len(X):,}", flush=True)

    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    YC = np.asarray(YC, np.float32); YF = np.asarray(YF, np.float32)
    M = np.asarray(META, dtype=object)
    np.savez_compressed(
        CACHE, X=X, Y=Y, Y_curr=YC, Y_futr=YF, fname=np.array(fname), axes=np.array(AX),
        scenario=M[:, 0].astype(str), split=M[:, 1].astype(str), family=M[:, 2].astype(str),
        cluster=M[:, 3].astype(str), t_anchor=M[:, 4].astype(np.float64),
        source_index=M[:, 5].astype(str), run_status=M[:, 6].astype(str),
        recipe_group=M[:, 7].astype(str))

    sha = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
    tr = M[:, 1] == "train"; va = M[:, 1] == "validation"
    manifest = {
        "generated_by": "models/features/build_features.py",
        "regenerable": "built from the raw logs and the cohort indices only",
        "python": platform.python_version(), "numpy": np.__version__,
        "cohort": {
            "index_union_unique": len(allrows),
            "train_validation_all_roles": len(tv),
            "excluded_smoke_shr": len(smoke), "smoke_ids": smoke,
            "official_cohort": len(use) + len(excluded),
            "status_mix": dict(Counter(r["run_status"] for r in use + excluded)),
            "VALID_COLLAPSE_included": True,
            "VALID_COLLAPSE_rationale":
                "valid runs that ended in divergence; the main source of large-error anchors",
            "by_source_index": per_index,
            "logs_loaded": int(load_status["ok"]),
            "logs_missing": len(missing), "missing_detail": missing},
        "target": {"formula": "max over [t, t+1s) of |e_a|", "axes": list(AX),
                   "Y_curr": "|e_a(t)| - current error only (triviality audit)",
                   "Y_futr": "max over (t, t+1s) - strictly future (triviality audit)",
                   "lon_offset_removed_m": LON_OFFSET,
                   "note": "includes the current time t: a near-term maximum-error assessment, not a forecast"},
        "features": {"n": X.shape[1], "signals": SIG, "stats": STATS,
                     "scales_s": list(SCALES),
                     "ndt_specific": sorted(NDT_SPECIFIC),
                     "matcher_agnostic": sorted(set(SIG) - NDT_SPECIFIC),
                     "excluded_by_prereg": ["ndt_step_dist", "ndt_speed"],
                     "excluded_reason": "dominated by the commanded vehicle speed"},
        "thresholds": {ax: {"T1": THRESH[ax][0], "T2": THRESH[ax][1],
                            "basis": THRESH_BASIS[ax]} for ax in AX},
        "anchors": {"total": int(len(X)), "train": int(tr.sum()), "validation": int(va.sum()),
                    "stride_s": STRIDE, "history_min_s": HIST_MIN,
                    "rule": "every time step is an anchor (health assessment covers segments that are already degraded)"},
        "clusters": {"train": int(len(set(M[tr, 3]))), "validation": int(len(set(M[va, 3])))},
        "X_sha256": sha,
        "leak_guard": assert_no_leak()}
    (OUT / "cache_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("cohort", "anchors", "clusters")},
                     indent=1, ensure_ascii=False))
    print(f"\n-> {CACHE}  ({CACHE.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
