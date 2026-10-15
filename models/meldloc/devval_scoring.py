#!/usr/bin/env python3
"""Development-validation scoring: every registered metric for MELD-Loc and the baselines.

Metric-output correspondence (fixed in advance):
  global rejection and MAE = median estimate / mid-band Spearman = R / large failures = P(Y>T2), P(Y>3W)
Distributional metrics (twIBS, monotonicity violations, coverage) are compared with the distributional baselines only.
(The earlier step-function quantile reconstruction was dropped as a distributional baseline;
Q-multi in distributional_baselines.py replaces it.)

All learned baselines use the same 120-d features and the same training split:
  P_asinh (L1 regression in asinh space), C2 (classifier at T1), TP-GBDT (24-d matcher-score features), logistic,
  Q-GBDT (q50/q90 quantiles), K-indep (independent classifiers on the 9-point grid)
Verdicts: paired cluster bootstrap 95% CI - WIN (CI excludes 0) / POINT_WIN (point estimate better,
CI includes 0) / TIE / LOSS (point estimate worse).


"""
from __future__ import annotations

import argparse
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
OUT = HERE / "results/development/devval_scoring.json"
N_BOOT = 600


def verdict(diff_point, ci, higher_better=True, eps=1e-4):
    lo, hi = ci
    d = diff_point if higher_better else -diff_point
    lo_, hi_ = (lo, hi) if higher_better else (-hi, -lo)
    if np.isfinite(lo_) and lo_ > 0:
        return "WIN"
    if np.isfinite(hi_) and hi_ < 0:
        return "LOSS_CI"
    if d > eps:
        return "POINT_WIN"
    if d < -eps:
        return "LOSS_POINT"
    return "TIE"


def main():
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default="lon,lat,yaw")
    ap.add_argument("--cache", default=str(CACHE))
    a = ap.parse_args()

    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    Z = np.load(a.cache, allow_pickle=True)
    X = Z["X"].astype(np.float64)
    Y = Z["Y"].astype(np.float64)
    fname = list(Z["fname"].astype(str))
    AXN = list(Z["axes"].astype(str))
    split = Z["split"].astype(str)
    fam = Z["family"].astype(str)
    clus = Z["cluster"].astype(str)
    grp = Z["recipe_group"].astype(str) if "recipe_group" in Z else clus
    scen = Z["scenario"].astype(str)
    tr, va = split == "train", split == "validation"
    tp_cols = np.array([f.startswith("transform_probability|") for f in fname])
    gbdt = mc.GBDT_DEFAULT

    res = {"status": "DEVELOPMENT_VALIDATION_SCORING",
           "revision": "R integration grid: 65 uniform points on [T1,3W] (end points included); "
                       "monotonicity violations measured on raw curves",
           "version_hash": mc.version_hash(), "n_boot": N_BOOT, "axes": {}}

    for ax in (x.strip() for x in a.axes.split(",")):
        ai = AXN.index(ax)
        y = Y[:, ai]
        yv, fv, cv, sv = y[va], fam[va], grp[va], scen[va]
        t1, t2, w3 = mc.T1[ax], mc.T2[ax], mc.W3[ax]
        model = mc.MeldLoc.load(MODEL_DIR / f"meldloc_{ax}.joblib")
        out = model.outputs(X[va])
        S, tg = out["S"], model.t_grid
        at = float(np.arcsinh(model.t_max))
        log(f"{ax}: MELD outputs ready (lambda={model.lam})")

        # ---- baselines (same features, same training split) ----
        P_as = np.sinh(lgb.LGBMRegressor(objective="l1", **gbdt)
                       .fit(X[tr], np.arcsinh(y[tr])).predict(X[va]))
        C2 = lgb.LGBMClassifier(is_unbalance=True, **gbdt).fit(
            X[tr], (y[tr] > t1).astype(int)).predict_proba(X[va])[:, 1]
        TPC = lgb.LGBMClassifier(is_unbalance=True, **gbdt).fit(
            X[tr][:, tp_cols], (y[tr] > t1).astype(int)
        ).predict_proba(X[va][:, tp_cols])[:, 1]
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X[tr])
        LOGR = LogisticRegression(max_iter=2000).fit(
            sc.transform(X[tr]), (y[tr] > t1).astype(int)
        ).predict_proba(sc.transform(X[va]))[:, 1]
        q50b = np.sinh(lgb.LGBMRegressor(objective="quantile", alpha=0.5, **gbdt)
                       .fit(X[tr], np.arcsinh(y[tr])).predict(X[va]))
        q90b = np.sinh(lgb.LGBMRegressor(objective="quantile", alpha=0.9, **gbdt)
                       .fit(X[tr], np.arcsinh(y[tr])).predict(X[va]))
        tg9 = np.sinh(np.linspace(0, at, 9))
        S_ind = np.empty((int(va.sum()), 9), np.float32)
        for j, tv in enumerate(tg9):
            S_ind[:, j] = lgb.LGBMClassifier(is_unbalance=True, **gbdt).fit(
                X[tr], (y[tr] > tv).astype(int)).predict_proba(X[va])[:, 1]
        log(f"{ax}: baselines trained")

        A = {"lambda": model.lam}
        mid = (yv > t1) & (yv <= w3)
        gen = ~np.isin(fv, list(mc.CONTROL))
        cm = cv[mid]

        # -- mid-band Spearman: pooled and family-macro reported separately --
        # the registered primary result is the family-macro value over general-degradation families;
        # the pooled value is a reference figure only.
        sp = lambda s, ii: mc.spearman(s[ii], yv[mid][ii])
        R = out["R"]
        ci_p = mc.paired_group_ci(sp, R[mid], P_as[mid], cm, N_BOOT)
        ci_c = mc.paired_group_ci(sp, R[mid], C2[mid], cm, N_BOOT)
        d_p = mc.spearman(R[mid], yv[mid]) - mc.spearman(P_as[mid], yv[mid])
        d_c = mc.spearman(R[mid], yv[mid]) - mc.spearman(C2[mid], yv[mid])
        A["mid_spearman_pooled"] = {
            "note": "reference figure, not the registered primary result",
            "MELD_R": round(mc.spearman(R[mid], yv[mid]), 4),
            "P_asinh": round(mc.spearman(P_as[mid], yv[mid]), 4),
            "C2": round(mc.spearman(C2[mid], yv[mid]), 4),
            "vs_P_CI": [round(v, 4) for v in ci_p],
            "vs_C2_CI": [round(v, 4) for v in ci_c],
            "verdict_vs_P": verdict(d_p, ci_p), "verdict_vs_C2": verdict(d_c, ci_c)}
        # family-macro (registered primary form, paired CI)
        gm = mid & gen
        fam_gm = fv[gm]
        ok_f = [f for f in sorted(set(fam_gm)) if (fam_gm == f).sum() >= 20]
        def macro(s, ii):
            vals = []
            for f in ok_f:
                m = fam_gm[ii] == f
                if m.sum() >= 10:
                    vals.append(mc.spearman(s[ii][m], yv[gm][ii][m]))
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")
        allg = np.arange(int(gm.sum()))
        ci_gm = mc.paired_group_ci(macro, R[gm], P_as[gm], cv[gm], N_BOOT)
        d_gm = macro(R[gm], allg) - macro(P_as[gm], allg)
        # per-family anchor counts (cells with n<50 give unstable Spearman values)
        per = {f: {"MELD": round(mc.spearman(R[gm][fam_gm == f], yv[gm][fam_gm == f]), 3),
                   "P_asinh": round(mc.spearman(P_as[gm][fam_gm == f],
                                                yv[gm][fam_gm == f]), 3),
                   "n_anchors": int((fam_gm == f).sum()),
                   "n_groups": int(len(set(cv[gm][fam_gm == f])))}
               for f in ok_f}
        A["mid_spearman_general_macro"] = {
            "note": "registered primary result (family-macro over general-degradation families)",
            "MELD_R": round(macro(R[gm], allg), 4),
            "P_asinh": round(macro(P_as[gm], allg), 4),
            "vs_P_CI": [round(v, 4) for v in ci_gm],
            "verdict_vs_P": verdict(d_gm, ci_gm),
            "n_groups_mid": int(len(set(cv[gm]))),
            "families": per}

        # -- global rejection at 10% (median estimate) and MAE --
        def top10(s, ii):
            o = np.argsort(-s[ii], kind="mergesort")
            k = int(len(ii) * 0.1)
            return float(yv[ii][o][k:].mean())
        e_med = out["e_med"]
        allid = np.arange(len(yv))
        ci_t = mc.paired_group_ci(top10, e_med, P_as, cv, N_BOOT)
        d_t = top10(e_med, allid) - top10(P_as, allid)
        A["top10_retained_mean_err"] = {
            "MELD_e_med": round(top10(e_med, allid), 4),
            "P_asinh": round(top10(P_as, allid), 4),
            "CI": [round(v, 4) for v in ci_t],
            "verdict": verdict(d_t, ci_t, higher_better=False)}
        mae = lambda s, ii: float(np.abs(s[ii] - yv[ii]).mean())
        ci_m = mc.paired_group_ci(mae, e_med, q50b, cv, N_BOOT)
        d_m = mae(e_med, allid) - mae(q50b, allid)
        A["mae_median"] = {"MELD_e_med": round(mae(e_med, allid), 4),
                           "Q_gbdt_q50": round(mae(q50b, allid), 4),
                           "P_asinh": round(mae(P_as, allid), 4),
                           "CI_vs_q50": [round(v, 4) for v in ci_m],
                           "verdict_vs_q50": verdict(d_m, ci_m, higher_better=False)}

        # -- discrimination: AUROC at T1 / T2 / 3W --
        lab1, lab2, lab3 = (yv > t1).astype(int), (yv > t2).astype(int), (yv > w3).astype(int)
        p1 = out["p_T1"]
        au = lambda s, ii: mc.auroc(s[ii], lab1[ii])
        ci_a = mc.paired_group_ci(au, p1, C2, cv, N_BOOT)
        d_a = mc.auroc(p1, lab1) - mc.auroc(C2, lab1)
        A["auroc_T1"] = {"MELD": round(mc.auroc(p1, lab1), 5),
                         "C2": round(mc.auroc(C2, lab1), 5),
                         "TP_GBDT": round(mc.auroc(TPC, lab1), 5),
                         "logistic": round(mc.auroc(LOGR, lab1), 5),
                         "vs_C2_CI": [round(v, 5) for v in ci_a],
                         "verdict_vs_C2": verdict(d_a, ci_a)}
        A["auroc_T2"] = {"MELD": round(mc.auroc(out["p_T2"], lab2), 5),
                         "P_asinh_as_score": round(mc.auroc(P_as, lab2), 5),
                         "C2_as_score": round(mc.auroc(C2, lab2), 5)}
        A["auroc_3W"] = {"MELD": round(mc.auroc(out["p_3W"], lab3), 5),
                         "P_asinh_as_score": round(mc.auroc(P_as, lab3), 5)}

        # -- distributional metrics (distributional baselines only) --
        w_t = np.where((tg9 >= t1) & (tg9 <= w3), 2.0, 1.0)
        lab_t = (yv[:, None] > tg9[None, :]).astype(float)
        S9 = np.empty_like(S_ind)
        for j, tv in enumerate(tg9):
            S9[:, j] = mc.prob_at(S, tg, tv)
        twbs = lambda Sm: float((w_t[None, :] * (Sm - lab_t) ** 2).mean())
        d_ib = twbs(S9) - twbs(S_ind)
        ib = lambda Sm, ii: float((w_t[None, :] * (Sm[ii] - lab_t[ii]) ** 2).mean())
        ci_ib = mc.paired_group_ci(ib, S9, S_ind, cv, N_BOOT)
        A["tw_integrated_brier"] = {
            "MELD": round(twbs(S9), 5), "K_indep": round(twbs(S_ind), 5),
            "baseline_note": "the step-function quantile reconstruction is not used as a baseline "
                             "(replaced by Q-multi)",
            "vs_K_CI": [round(v, 5) for v in ci_ib],
            "verdict_vs_K": verdict(d_ib, ci_ib, higher_better=False)}
        A["monotone_violation"] = {
            "MELD_measured_pre_projection": round(float(out["viol"]), 6),
            "K_indep": round(float((np.diff(S_ind, axis=1) > 1e-6).mean()), 4),
            "note": "MELD values are measured on raw dev-val curves (adjacent-pair increase rate before projection)"}

        # -- 90% bound coverage (per-run criterion: |deviation| <= 3 pp) --
        q90c = out["q90_conf"]
        cov = float((yv <= q90c).mean())
        cov_q = float((yv <= q90b).mean())
        per_scen = []
        for s in sorted(set(sv)):
            m = sv == s
            if m.sum() >= 50:
                per_scen.append(float((yv[m] <= q90c[m]).mean()))
        scen_err = abs(float(np.mean(per_scen)) - 0.9) if per_scen else float("nan")
        per_famc = {f: round(float((yv[fv == f] <= q90c[fv == f]).mean()), 4)
                    for f in sorted(set(fv))}
        A["q90_upper_coverage"] = {
            "MELD_conformal": round(cov, 4), "Q_gbdt_q90": round(cov_q, 4),
            "scenario_mean_abs_err": round(scen_err, 4),
            "criterion_3pp": bool(scen_err <= 0.03),
            "per_family": per_famc,
            "verdict": "WIN" if abs(cov - 0.9) < abs(cov_q - 0.9) else "LOSS_POINT"}

        # -- scoreboard --
        verd = {k: v for k, v in [
            ("mid_macro_vs_P_(primary)", A["mid_spearman_general_macro"]["verdict_vs_P"]),
            ("mid_pooled_vs_P_(reference)", A["mid_spearman_pooled"]["verdict_vs_P"]),
            ("mid_pooled_vs_C2_(reference)", A["mid_spearman_pooled"]["verdict_vs_C2"]),
            ("top10", A["top10_retained_mean_err"]["verdict"]),
            ("mae_vs_q50", A["mae_median"]["verdict_vs_q50"]),
            ("auroc_T1_vs_C2", A["auroc_T1"]["verdict_vs_C2"]),
            ("twIBS_vs_K", A["tw_integrated_brier"]["verdict_vs_K"]),
            ("q90_coverage", A["q90_upper_coverage"]["verdict"])]}
        A["scoreboard"] = verd
        A["losses"] = sorted(k for k, v in verd.items() if v.startswith("LOSS"))
        res["axes"][ax] = A
        log(f"{ax}: scoreboard {json.dumps(verd, ensure_ascii=False)}")

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")
    print(json.dumps({ax: res["axes"][ax]["scoreboard"] for ax in res["axes"]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
