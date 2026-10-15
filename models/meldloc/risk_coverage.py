#!/usr/bin/env python3
"""Risk-coverage curve (AURC) on the development-validation split.

Computation only: frozen model, no new data, no holdout access.

Specification (fixed in advance):
  - ranking output = the median estimate (the registered output for risk-coverage,
    top-k rejection and MAE)
  - rejection grid 0-50 % in 1 % steps (51 points)
  - risk = mean realized |error| of the retained anchors
  - AURC = mean retained risk over the grid; point values at 5/10/20/30 %
    (10 % doubles as a check against the top-10% figure of meldloc_eval)
  - comparison = P_asinh
  - paired recipe-group bootstrap 95 % CI, three axes
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
OUT = HERE / "results/development/risk_coverage.json"
N_BOOT = 600
GRID = np.arange(0, 51)          # rejection rate in %
POINTS = (5, 10, 20, 30)


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

    res = {"status": "REGISTERED_METRIC_COMPLETION (rev.7) — dev, NON_CONFIRMATORY",
           "spec": "rank=median estimate / rejection 0-50% in 1% steps / risk=mean realized "
                   "|error| of retained anchors / AURC=grid mean / comparison=P_asinh / group bootstrap CI",
           "n_boot": N_BOOT, "axes": {}}

    for ax in ("lon", "lat", "yaw"):
        ai = AXN.index(ax)
        y = Y[:, ai]
        yv, cv = y[va], grp[va]
        m = mc.MeldLoc.load(MODEL_DIR / f"meldloc_{ax}.joblib")
        e_med = m.outputs(X[va])["e_med"]
        P = np.sinh(lgb.LGBMRegressor(objective="l1", **mc.GBDT_DEFAULT)
                    .fit(X[tr], np.arcsinh(y[tr])).predict(X[va]))
        log(f"{ax}: outputs and baseline ready")

        def curve(s, ii):
            ys = yv[ii][np.argsort(-s[ii], kind="mergesort")]
            n = len(ys)
            cs = np.concatenate([[0.0], np.cumsum(ys)])
            ks = np.minimum((n * GRID / 100).astype(int), n - 1)
            return (cs[n] - cs[ks]) / (n - ks)

        aurc = lambda s, ii: float(curve(s, ii).mean())
        allid = np.arange(len(yv))
        cm, cp = curve(e_med, allid), curve(P, allid)
        d = aurc(e_med, allid) - aurc(P, allid)
        ci = mc.paired_group_ci(aurc, e_med, P, cv, N_BOOT)
        verdict = ("WIN" if ci[1] < 0 else "LOSS_CI" if ci[0] > 0 else
                   "POINT_WIN" if d < -1e-5 else
                   "LOSS_POINT" if d > 1e-5 else "TIE")
        res["axes"][ax] = {
            "AURC": {"MELD_e_med": round(float(cm.mean()), 4),
                     "P_asinh": round(float(cp.mean()), 4)},
            "diff_MELD_minus_P": round(d, 4),
            "CI": [round(v, 4) for v in ci],
            "verdict(lower_is_better)": verdict,
            "points": {f"reject_{p}%": {"MELD": round(float(cm[p]), 4),
                                        "P_asinh": round(float(cp[p]), 4)}
                       for p in POINTS},
            "top10_consistency_check": {
                "MELD(from_eval)": None, "reject_10%_here": round(float(cm[10]), 4)}}
        log(f"{ax}: AURC MELD {cm.mean():.4f} vs P {cp.mean():.4f} "
            f"diff {d:+.4f} CI {ci} => {verdict}")

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
