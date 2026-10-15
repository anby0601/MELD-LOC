#!/usr/bin/env python3
"""Reference methods without learning: BL-3sigma / BL-TP / B0
+ paired AUROC@T1 CIs for TP-GBDT and the logistic classifier.

The learning-free rules are not subject to the "same features, same folds" re-training rule.
Specification:
  BL-3sigma: the configured constant bound of the evaluated Autoware stack, lon/lat 0.45 m, yaw 4.295 deg
         (covariance_estimation_type=0, σ_xy 0.15 m·σ_yaw 1.432°, scale 3.0)
         coverage overall / by regime (normal y<=T1, mid T1<y<=3W, divergent y>3W) /
         by family, in the same table layout as the MELD bound.
  BL-TP: learning-free threshold rule, score = -transform_probability|2s|last.
  B0   : input-free reference from the training-set statistics (estimate = median, q90 = q90,
         S(t) = empirical exceedance function); AUROC with a constant score is exactly 0.5.
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
OUT = HERE / "results/baselines/reference_methods.json"
N_BOOT = 600
BL3S = {"lon": 0.45, "lat": 0.45, "yaw": 4.295}


def main():
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    Z = np.load(CACHE, allow_pickle=True)
    X = Z["X"].astype(np.float64)
    Y = Z["Y"].astype(np.float64)
    fname = list(Z["fname"].astype(str))
    AXN = list(Z["axes"].astype(str))
    split = Z["split"].astype(str)
    fam = Z["family"].astype(str)
    grp = Z["recipe_group"].astype(str)
    tr, va = split == "train", split == "validation"
    tp_cols = np.array([f.startswith("transform_probability|") for f in fname])
    tp_last = fname.index("transform_probability|2s|last")
    gbdt = mc.GBDT_DEFAULT

    res = {"status": "REGISTERED_BASELINE_COMPLETION (rev.9) — dev, NON_CONFIRMATORY",
           "naming": "BL-3sigma and BL-TP (reference methods)",
           "axes": {}}

    for ax in ("lon", "lat", "yaw"):
        ai = AXN.index(ax)
        y = Y[:, ai]
        yv, fv, cv = y[va], fam[va], grp[va]
        t1, t2, w3 = mc.T1[ax], mc.T2[ax], mc.W3[ax]
        M = mc.MeldLoc.load(MODEL_DIR / f"meldloc_{ax}.joblib")
        out = M.outputs(X[va])
        qM = out["q90_conf"]
        lab1 = (yv > t1).astype(int)
        A = {}

        # ---- BL-3sigma: coverage of the constant bound ----
        c = BL3S[ax]
        band = np.where(yv <= t1, "normal", np.where(yv <= w3, "middle", "divergent"))
        cov = lambda q, m_: (round(float((yv[m_] <= (q[m_] if hasattr(q, "__len__")
                                                     else q)).mean()), 4)
                             if m_.sum() > 0 else None)
        allm = np.ones(len(yv), bool)
        A["BL_3sigma"] = {
            "constant_bound": c,
            "coverage": {
                "overall": {"BL_3sigma": cov(np.full(len(yv), c), allm),
                            "MELD_conf": cov(qM, allm)},
                "by_band": {b: {"n": int((band == b).sum()),
                                "BL_3sigma": cov(np.full(len(yv), c), band == b),
                                "MELD_conf": cov(qM, band == b)}
                            for b in ("normal", "middle", "divergent")},
                "by_family": {f: {"BL_3sigma": cov(np.full(len(yv), c), fv == f),
                                  "MELD_conf": cov(qM, fv == f)}
                              for f in sorted(set(fv))}}}

        # ---- BL-TP: learning-free threshold rule (score = -TP last) ----
        s_tp = -X[va][:, tp_last]
        A["BL_TP_rule"] = {
            "note": "learning-free threshold rule (distinct from the learned TP-GBDT)",
            "auroc_T1": round(mc.auroc(s_tp, lab1), 5),
            "auroc_T2": round(mc.auroc(s_tp, (yv > t2).astype(int)), 5)}

        # ---- B0: input-free training-set statistics ----
        e0 = float(np.median(y[tr]))
        q0 = float(np.quantile(y[tr], 0.9))
        at = float(np.arcsinh(M.t_max))
        tg9 = np.sinh(np.linspace(0, at, 9))
        S0 = np.array([float((y[tr] > tv).mean()) for tv in tg9])
        w_t = np.where((tg9 >= t1) & (tg9 <= w3), 2.0, 1.0)
        lab_t = (yv[:, None] > tg9[None, :]).astype(float)
        S0m = np.tile(S0, (len(yv), 1))
        S9M = np.stack([mc.prob_at(out["S"], M.t_grid, tv) for tv in tg9], axis=1)
        ib = lambda Sm: float((w_t[None, :] * (Sm - lab_t) ** 2).mean())
        A["B0_climatology"] = {
            "definition": "input-free empirical quantiles / exceedance function of the training set",
            "mae": round(float(np.abs(e0 - yv).mean()), 4),
            "MELD_mae": round(float(np.abs(out["e_med"] - yv).mean()), 4),
            "auroc_T1": 0.5,
            "twIBS": round(ib(S0m), 5), "MELD_twIBS": round(ib(S9M), 5),
            "q90_coverage": round(float((yv <= q0).mean()), 4),
            "q90_constant": round(q0, 3),
            "top10_note": "constant score -> random rejection; retained risk = overall mean "
                          f"{round(float(yv.mean()), 4)}"}

        # ---- paired AUROC@T1 CIs (TP-GBDT, logistic) ----
        TPC = lgb.LGBMClassifier(is_unbalance=True, **gbdt).fit(
            X[tr][:, tp_cols], (y[tr] > t1).astype(int)
        ).predict_proba(X[va][:, tp_cols])[:, 1]
        sc = StandardScaler().fit(X[tr])
        LOGR = LogisticRegression(max_iter=2000).fit(
            sc.transform(X[tr]), (y[tr] > t1).astype(int)
        ).predict_proba(sc.transform(X[va]))[:, 1]
        p1 = out["p_T1"]
        au = lambda s, ii: mc.auroc(s[ii], lab1[ii])
        ci_tp = mc.paired_group_ci(au, p1, TPC, cv, N_BOOT)
        ci_lg = mc.paired_group_ci(au, p1, LOGR, cv, N_BOOT)
        vd = lambda d, ci: ("WIN" if ci[0] > 0 else "LOSS_CI" if ci[1] < 0 else
                            "POINT_WIN" if d > 1e-4 else
                            "LOSS_POINT" if d < -1e-4 else "TIE")
        A["auroc_T1_missing_CIs"] = {
            "MELD": round(mc.auroc(p1, lab1), 5),
            "TP_GBDT": round(mc.auroc(TPC, lab1), 5),
            "logistic": round(mc.auroc(LOGR, lab1), 5),
            "vs_TP_GBDT": {"CI": [round(v, 5) for v in ci_tp],
                           "verdict": vd(mc.auroc(p1, lab1) - mc.auroc(TPC, lab1),
                                         ci_tp)},
            "vs_logistic": {"CI": [round(v, 5) for v in ci_lg],
                            "verdict": vd(mc.auroc(p1, lab1) - mc.auroc(LOGR, lab1),
                                          ci_lg)}}
        res["axes"][ax] = A
        log(f"{ax}: done (BL-3sigma overall {A['BL_3sigma']['coverage']['overall']} / "
            f"BL-TP auroc {A['BL_TP_rule']['auroc_T1']})")

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
