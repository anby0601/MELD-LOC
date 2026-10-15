#!/usr/bin/env python3
"""Distributional baselines Q-multi and K-indep, computation only.


  1) Q-multi: the exceedance curve reconstructed from nine quantile regressions
     (the registered quantile-reconstruction baseline).
  2) K-indep: independent classifiers; the "calibrated" variant applies the same isotonic
     calibration as MELD (is_unbalance=False) to remove the calibration asymmetry;
     the uncalibrated variant is kept for reference.

-- specification (fixed in advance) --
Q-multi:
  - nine quantile levels q10..q90, LightGBM quantile objective, asinh label space,
    120-d features, same training split, same GBDT budget.
  - crossing: per-anchor np.maximum.accumulate in asinh space (non-decreasing quantiles).
    Crossing rates are reported on the raw values: (a) per anchor (any crossing), (b) per adjacent pair.
  - predicted asinh quantiles are clipped at 0 (y = |error| >= 0).
  - curve: piecewise-linear in arcsinh(t) through the knots (0, 1.0), (q10, 0.9), ...,
    (q90, 0.1); beyond q90 the last slope (q80->q90) is extrapolated and clipped to [0,1];
    a segment shorter than 1e-9 gives the constant 0.1; the x knots are made strictly increasing by 1e-9 steps.
  - monotonicity violations: before projection = adjacent-pair crossing rate (b); after = rate of
    diff > 1e-6 on the tg9 curve (same rule as MELD and K-indep).
K-indep calibrated variant:
  - is_unbalance=False, thresholds = the 9-point grid tg9.
  - isotonic: natural dev-val distribution, the 9 thresholds pooled, equal group weights
    (gw_va repeated 9 times); the same procedure as MELD step 7, with the same
    fit-on-dev optimism as MELD (the point is symmetric treatment).
twIBS: tg9 grid, mid-band weight 2, same labels. Four curves:
  MELD / K-indep (uncalibrated) / K-indep (calibrated) / Q-multi.
  paired recipe-group bootstrap 95% CI: MELD - K-indep(cal), MELD - Q-multi.
Also: Q-multi q90 coverage (reference).
──────────────────────────────────────────────────────────────────────
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
OUT = HERE / "results/baselines/distributional_baselines.json"
N_BOOT = 600
Q_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # fixed


def qmulti_curve(aq_sorted, atg):
    """Per-anchor nine asinh quantiles -> S on the tg9 grid (see the specification above)."""
    n = len(aq_sorted)
    S = np.empty((n, len(atg)), np.float32)
    y_kn = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    for i in range(n):
        xp = np.concatenate([[0.0], aq_sorted[i]])
        xp = xp + np.arange(10) * 1e-9          # strictly increasing knots
        s = np.interp(atg, xp, y_kn)
        seg = xp[-1] - xp[-2]
        beyond = atg > xp[-1]
        if beyond.any():
            if seg < 1e-9:
                s[beyond] = 0.1
            else:
                slope = (0.1 - 0.2) / seg
                s[beyond] = np.clip(0.1 + slope * (atg[beyond] - xp[-1]), 0.0, 1.0)
        S[i] = s
    return S


def main():
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    Z = np.load(CACHE, allow_pickle=True)
    X = Z["X"].astype(np.float64)
    Y = Z["Y"].astype(np.float64)
    AXN = list(Z["axes"].astype(str))
    split = Z["split"].astype(str)
    grp = Z["recipe_group"].astype(str)
    tr, va = split == "train", split == "validation"
    gw_va = mc.group_equal_weights(grp[va])
    gbdt = mc.GBDT_DEFAULT

    res = {"status": "DISTRIBUTIONAL_BASELINES (dev-val scoring)",
           "spec": "Q-multi q10..q90 in asinh space, maximum.accumulate, arcsinh piecewise-linear, "
                   "clipped extrapolation / K-indep calibrated: is_unbalance=False + dev isotonic "
                   "(equal group weights, same procedure as MELD)",
           "n_boot": N_BOOT, "axes": {}}

    for ax in ("lon", "lat", "yaw"):
        ai = AXN.index(ax)
        y = Y[:, ai]
        yv, cv = y[va], grp[va]
        t1, w3 = mc.T1[ax], mc.W3[ax]
        m = mc.MeldLoc.load(MODEL_DIR / f"meldloc_{ax}.joblib")
        at = float(np.arcsinh(m.t_max))
        tg9 = np.sinh(np.linspace(0, at, 9))
        atg9 = np.arcsinh(tg9)
        w_t = np.where((tg9 >= t1) & (tg9 <= w3), 2.0, 1.0)
        lab_t = (yv[:, None] > tg9[None, :]).astype(float)
        twbs = lambda Sm: float((w_t[None, :] * (Sm - lab_t) ** 2).mean())
        ib = lambda Sm, ii: float((w_t[None, :] * (Sm[ii] - lab_t[ii]) ** 2).mean())

        # MELD S on the 9-point grid (as in meldloc_eval)
        S_meld = m.curve(X[va])
        S9 = np.stack([mc.prob_at(S_meld, m.t_grid, tv) for tv in tg9], axis=1)

        # K-indep uncalibrated (is_unbalance=True): reproduces meldloc_eval (same seed, same numbers)
        def k_indep(unbal):
            S = np.empty((int(va.sum()), 9), np.float32)
            for j, tv in enumerate(tg9):
                S[:, j] = lgb.LGBMClassifier(is_unbalance=unbal, **gbdt).fit(
                    X[tr], (y[tr] > tv).astype(int)).predict_proba(X[va])[:, 1]
            return S
        S_kc = k_indep(True)
        # K-indep calibrated: is_unbalance=False + dev isotonic (MELD step 7)
        S_kf_raw = k_indep(False)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(S_kf_raw.reshape(-1), lab_t.reshape(-1),
                sample_weight=np.repeat(gw_va, 9))
        S_kf = iso.predict(S_kf_raw.reshape(-1)).reshape(S_kf_raw.shape).astype(np.float32)
        log(f"{ax}: K-indep uncalibrated/calibrated done")

        # Q-multi
        AQ = np.empty((int(va.sum()), 9))
        for j, q in enumerate(Q_LEVELS):
            AQ[:, j] = lgb.LGBMRegressor(objective="quantile", alpha=q, **gbdt).fit(
                X[tr], np.arcsinh(y[tr])).predict(X[va])
        AQ = np.maximum(AQ, 0.0)                       # clip at 0
        cross_pair = float((np.diff(AQ, axis=1) < 0).mean())
        cross_anchor = float((np.diff(AQ, axis=1) < 0).any(axis=1).mean())
        AQs = np.maximum.accumulate(AQ, axis=1)
        S_q = qmulti_curve(AQs, atg9)
        viol_q_post = float((np.diff(S_q, axis=1) > 1e-6).mean())
        cov_q90 = float((yv <= np.sinh(AQs[:, -1])).mean())
        log(f"{ax}: Q-multi done (crossing anchors {cross_anchor:.3f} / pairs {cross_pair:.3f})")

        ci_kf = mc.paired_group_ci(ib, S9, S_kf, cv, N_BOOT)
        ci_q = mc.paired_group_ci(ib, S9, S_q, cv, N_BOOT)
        d_kf, d_q = twbs(S9) - twbs(S_kf), twbs(S9) - twbs(S_q)
        vd = lambda d, ci: ("WIN" if ci[1] < 0 else
                            "LOSS_CI" if ci[0] > 0 else
                            "POINT_WIN" if d < -1e-5 else
                            "LOSS_POINT" if d > 1e-5 else "TIE")
        res["axes"][ax] = {
            "twIBS": {"MELD": round(twbs(S9), 5),
                      "K_indep_current(is_unbalance,uncalibrated)": round(twbs(S_kc), 5),
                      "K_indep_fixed(+isotonic)": round(twbs(S_kf), 5),
                      "Q_multi": round(twbs(S_q), 5)},
            "MELD_vs_Kfixed": {"diff": round(d_kf, 5),
                               "CI": [round(v, 5) for v in ci_kf],
                               "verdict": vd(d_kf, ci_kf)},
            "MELD_vs_Qmulti": {"diff": round(d_q, 5),
                               "CI": [round(v, 5) for v in ci_q],
                               "verdict": vd(d_q, ci_q)},
            "qmulti_crossing": {"anchor_rate": round(cross_anchor, 4),
                                "pair_rate": round(cross_pair, 4)},
            "qmulti_monotone_violation": {"pre_sort(adjacent_pair_crossing_rate)": round(cross_pair, 4),
                                          "post_sort_curve": round(viol_q_post, 6)},
            "qmulti_q90_coverage": round(cov_q90, 4)}
        log(f"{ax}: twIBS MELD {twbs(S9):.5f} Kfix {twbs(S_kf):.5f} "
            f"Qmulti {twbs(S_q):.5f} | vsKfix {vd(d_kf, ci_kf)} vsQ {vd(d_q, ci_q)}")

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
