#!/usr/bin/env python3
"""Tightness of the 90% bound and inference cost on the development-validation split (descriptive).

Specification (fixed in advance):
  - MELD q90_conf vs Q-GBDT q90 (the same comparator as the coverage metric)
  - tightness = mean arcsinh(q90) (one-sided bound: width = bound size, lower is tighter)
    + raw median
  - combined score = pinball loss at tau=0.9 in asinh space:
      ρ(y, q) = 0.9·(ay−aq)⁺ + 0.1·(aq−ay)⁺,  ay=arcsinh(y), aq=arcsinh(q)
    (the proper score of a one-sided bound, instead of the two-sided Winkler score)
  - paired recipe-group bootstrap 95 % CI (informational)
  - inference cost: batch throughput over dev (all four outputs) + single-anchor latency
    (median of 20 repetitions)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "models/features"))
sys.path.insert(0, str(HERE / "models/meldloc"))
import meldloc_core as mc  # noqa: E402

CACHE = HERE / "results/features/development_features.npz"
MODEL_DIR = HERE / "models/frozen"
OUT = HERE / "results/development/bound_tightness.json"
N_BOOT = 600


def main():
    import lightgbm as lgb

    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    Z = np.load(CACHE, allow_pickle=True)
    X = Z["X"].astype(np.float64)
    Y = Z["Y"].astype(np.float64)
    AXN = list(Z["axes"].astype(str))
    split = Z["split"].astype(str)
    grp = Z["recipe_group"].astype(str)
    tr, va = split == "train", split == "validation"

    res = {"status": "DESCRIPTIVE (dev-val)",
           "n_boot": N_BOOT, "axes": {}, "inference_cost": {}}

    for ax in ("lon", "lat", "yaw"):
        ai = AXN.index(ax)
        y = Y[:, ai]
        yv, cv = y[va], grp[va]
        m = mc.MeldLoc.load(MODEL_DIR / f"meldloc_{ax}.joblib")

        t1 = time.time()
        out = m.outputs(X[va])
        batch_s = time.time() - t1
        qM = out["q90_conf"]
        qB = np.sinh(lgb.LGBMRegressor(objective="quantile", alpha=0.9,
                                       **mc.GBDT_DEFAULT)
                     .fit(X[tr], np.arcsinh(y[tr])).predict(X[va]))

        ay = np.arcsinh(yv)
        pin = lambda q, ii: float(np.mean(
            np.where(ay[ii] > np.arcsinh(q[ii]),
                     0.9 * (ay[ii] - np.arcsinh(q[ii])),
                     0.1 * (np.arcsinh(q[ii]) - ay[ii]))))
        allid = np.arange(len(yv))
        ci = mc.paired_group_ci(pin, qM, qB, cv, N_BOOT)
        res["axes"][ax] = {
            "coverage(repeated)": {"MELD_conf": round(float((yv <= qM).mean()), 4),
                                "Q_gbdt": round(float((yv <= qB).mean()), 4)},
            "tightness_mean_asinh(lower_is_tighter)": {
                "MELD_conf": round(float(np.arcsinh(qM).mean()), 4),
                "Q_gbdt": round(float(np.arcsinh(qB).mean()), 4)},
            "bound_raw_median_m_or_deg": {
                "MELD_conf": round(float(np.median(qM)), 3),
                "Q_gbdt": round(float(np.median(qB)), 3)},
            "pinball_tau0.9_asinh(lower_is_better)": {
                "MELD_conf": round(pin(qM, allid), 5),
                "Q_gbdt": round(pin(qB, allid), 5),
                "diff_M_minus_Q": round(pin(qM, allid) - pin(qB, allid), 5),
                "CI(informational)": [round(v, 5) for v in ci]}}
        log(f"{ax}: pinball M {pin(qM, allid):.5f} vs Q {pin(qB, allid):.5f} "
            f"| asinh width M {np.arcsinh(qM).mean():.3f} vs Q {np.arcsinh(qB).mean():.3f}")

        # inference cost (all four outputs)
        lat_ms = []
        x1 = X[va][:1]
        for _ in range(20):
            s = time.time()
            m.outputs(x1)
            lat_ms.append((time.time() - s) * 1000)
        res["inference_cost"][ax] = {
            "batch_dev_total_s": round(batch_s, 2),
            "batch_per_anchor_ms": round(batch_s / int(va.sum()) * 1000, 3),
            "single_anchor_ms_median(20_reps)": round(float(np.median(lat_ms)), 1)}
        log(f"{ax}: batch inference {batch_s:.1f}s ({batch_s/int(va.sum())*1000:.2f} ms/anchor), "
            f"single anchor {np.median(lat_ms):.0f} ms")

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
