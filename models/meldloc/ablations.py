#!/usr/bin/env python3
"""Registered ablations: no monotone constraint / lambda=0 / no family re-weighting / rank by P(Y>T1) / latest values only.

Each ablation is a separate re-training; the frozen model is not modified.
Internal labels: A1 = no monotone constraint, A3 = lambda=0, A4 = no family re-weighting, A7 = latest values only, A5 = rank by P(Y>T1).
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
QMULTI = HERE / "results/baselines/distributional_baselines.json"
OUT = HERE / "results/ablations/model_components.json"
N_BOOT = 600


def main():
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    Z = np.load(CACHE, allow_pickle=True)
    X = Z["X"].astype(np.float64)
    Y = Z["Y"].astype(np.float64)
    fname = list(Z["fname"].astype(str))
    AXN = list(Z["axes"].astype(str))
    split = Z["split"].astype(str)
    fam = Z["family"].astype(str)
    clus = Z["cluster"].astype(str)
    grp = Z["recipe_group"].astype(str)
    scen = Z["scenario"].astype(str)
    t_anchor = Z["t_anchor"].astype(np.float64)
    tr, va = split == "train", split == "validation"
    keys_tr = mc.anchor_keys(scen[tr], t_anchor[tr])
    keys_va = mc.anchor_keys(scen[va], t_anchor[va])
    fold_tr, _ = mc.cluster_folds(clus[tr])
    gw_va = mc.group_equal_weights(grp[va])
    gbdt = mc.GBDT_DEFAULT
    tp_all = np.array([f.startswith("transform_probability|") for f in fname])
    last_cols = np.array([f.endswith("|2s|last") for f in fname])
    qm = json.loads(QMULTI.read_text())

    res = {"status": "REGISTERED_ABLATIONS (dev-val; frozen assets unchanged)",
           "n_boot": N_BOOT, "axes": {}}

    for ax in ("lon", "lat", "yaw"):
        ai = AXN.index(ax)
        y = Y[:, ai]
        yv, fv, cv = y[va], fam[va], grp[va]
        t1, t2, w3 = mc.T1[ax], mc.T2[ax], mc.W3[ax]
        M = mc.MeldLoc.load(MODEL_DIR / f"meldloc_{ax}.joblib")
        t_max, tg = M.t_max, M.t_grid
        at = float(np.arcsinh(t_max))
        ycap = np.minimum(y, t_max)
        tg9 = np.sinh(np.linspace(0, at, 9))
        w_t = np.where((tg9 >= t1) & (tg9 <= w3), 2.0, 1.0)
        lab_t = (yv[:, None] > tg9[None, :]).astype(float)
        lab1 = (yv > t1).astype(int)
        allid = np.arange(len(yv))
        mid = (yv > t1) & (yv <= w3)
        gen = ~np.isin(fv, list(mc.CONTROL))
        gm = mid & gen
        fam_gm = fv[gm]
        ok_f = [f for f in sorted(set(fam_gm)) if (fam_gm == f).sum() >= 20]

        def macro(s, ii):
            vals = []
            for f in ok_f:
                mk = fam_gm[ii] == f
                if mk.sum() >= 10:
                    v = mc.spearman(s[ii][mk], yv[gm][ii][mk])
                    if np.isfinite(v):
                        vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")

        def top10(s, ii):
            o = np.argsort(-s[ii], kind="mergesort")
            k = int(len(ii) * 0.1)
            return float(yv[ii][o][k:].mean())

        mae = lambda s, ii: float(np.abs(s[ii] - yv[ii]).mean())
        au1 = lambda s, ii: mc.auroc(s[ii], lab1[ii])
        ib = lambda Sm, ii: float((w_t[None, :] * (Sm[ii] - lab_t[ii]) ** 2).mean())
        S9_of = lambda S: np.stack([mc.prob_at(S, tg, tv) for tv in tg9], axis=1)

        # frozen MELD reference outputs
        outM = M.outputs(X[va])
        S9M = S9_of(outM["S"])
        ref = {"R": outM["R"], "e": outM["e_med"], "p1": outM["p_T1"], "S9": S9M}

        def score(out, S9v, viol, cov):
            """Metric table plus paired CI against the frozen MELD model."""
            allg = np.arange(int(gm.sum()))
            sp_p = lambda s, ii: mc.spearman(s[ii], yv[mid][ii])
            r = {"mid_macro": round(macro(out["R"][gm], allg), 4),
                 "mid_pooled": round(mc.spearman(out["R"][mid], yv[mid]), 4),
                 "top10": round(top10(out["e_med"], allid), 4),
                 "mae": round(mae(out["e_med"], allid), 4),
                 "auroc_T1": round(mc.auroc(out["p_T1"], lab1), 5),
                 "twIBS": round(ib(S9v, allid), 5),
                 "viol_pre_projection": round(viol, 6),
                 "q90_cov": round(cov, 4)}
            r["vs_MELD_CI"] = {
                "mid_macro": [round(v, 4) for v in
                              mc.paired_group_ci(macro, out["R"][gm], ref["R"][gm],
                                                 cv[gm], N_BOOT)],
                "top10": [round(v, 4) for v in
                          mc.paired_group_ci(top10, out["e_med"], ref["e"], cv, N_BOOT)],
                "auroc_T1": [round(v, 5) for v in
                             mc.paired_group_ci(au1, out["p_T1"], ref["p1"], cv, N_BOOT)],
                "twIBS": [round(v, 5) for v in
                          mc.paired_group_ci(ib, S9v, ref["S9"], cv, N_BOOT)]}
            return r

        A = {"MELD_ref": {"mid_macro": round(macro(ref["R"][gm],
                                                   np.arange(int(gm.sum()))), 4),
                          "top10": round(top10(ref["e"], allid), 4),
                          "mae": round(mae(ref["e"], allid), 4),
                          "auroc_T1": round(mc.auroc(ref["p1"], lab1), 5),
                          "twIBS": round(ib(S9M, allid), 5),
                          "viol": round(outM["viol"], 6),
                          "q90_cov": round(float((yv <= outM["q90_conf"]).mean()), 4)},
             "Qmulti_twIBS_ref": qm["axes"][ax]["twIBS"]["Q_multi"]}

        # ---- shared: re-run the registered flow with one component removed ----
        def run_flow(mono=True, reweight=True, feat="full"):
            xa = (~tp_all) if feat == "full" else ((~tp_all) & last_cols)
            xs = tp_all if feat == "full" else (tp_all & last_cols)
            Xp, lab, w0, aidx = mc.make_pairs(X[tr][:, xa], ycap[tr], t_max,
                                              t1, t2, w3, keys_tr)
            fold_p = fold_tr[aidx]
            fam_p = fam[tr][aidx]
            kw_a = (dict(monotone_constraints=[0] * int(xa.sum()) + [-1],
                         monotone_constraints_method="advanced") if mono else {})
            kw_s = (dict(monotone_constraints=[0] * int(xs.sum()) + [-1],
                         monotone_constraints_method="advanced") if mono else {})

            def fit_F(w, mask=None):
                m_ = lgb.LGBMClassifier(**kw_a, **gbdt)
                s = slice(None) if mask is None else mask
                m_.fit(Xp[s], lab[s], sample_weight=w[s])
                return m_

            oof = np.zeros(len(lab))
            for k in range(5):
                m_ = fit_F(w0, fold_p != k)
                oof[fold_p == k] = m_.predict_proba(Xp[fold_p == k])[:, 1]
            if reweight:
                wf = mc.family_weights(oof, lab, fam_p)
                w_rw = w0 * np.array([wf[f] for f in fam_p])
                oof_w = np.zeros(len(lab))
                for k in range(5):
                    m_ = fit_F(w_rw, fold_p != k)
                    oof_w[fold_p == k] = m_.predict_proba(Xp[fold_p == k])[:, 1]
            else:
                wf = {f: 1.0 for f in sorted(set(fam_p))}
                w_rw, oof_w = w0, oof
            F = fit_F(w_rw)
            Xs_p = np.hstack([np.repeat(X[tr][:, xs], mc.K, axis=0),
                              Xp[:, -1:]]).astype(np.float32)
            G = lgb.LGBMClassifier(**kw_s, **gbdt)
            G.fit(Xs_p, lab, sample_weight=w_rw, init_score=mc.logit(oof_w))
            # lambda selection (dev-val, registered flow)
            Xp_v, lab_v, _, _ = mc.make_pairs(X[va][:, xa], np.minimum(yv, t_max),
                                              t_max, t1, t2, w3, keys_va)
            Xs_v = np.hstack([np.repeat(X[va][:, xs], mc.K, axis=0),
                              Xp_v[:, -1:]]).astype(np.float32)
            zf, zg = F.predict(Xp_v, raw_score=True), G.predict(Xs_v, raw_score=True)
            gw_p = np.repeat(gw_va, mc.K)
            eps = 1e-7
            ll = {}
            for lam in mc.LAM_GRID:
                p = mc.sigmoid(zf + lam * zg)
                l_ = -(lab_v * np.log(p + eps) + (1 - lab_v) * np.log(1 - p + eps))
                ll[lam] = float((l_ * gw_p).sum() / gw_p.sum())
            lam = min(ll, key=ll.get)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(mc.sigmoid(zf + lam * zg), lab_v, sample_weight=gw_p)
            mdl = mc.MeldLoc(ax, F, G, float(lam), iso, 0.0, t_max, tg, wf, xa, xs,
                             meta={"ablation": True})
            out = mdl.outputs(X[va])
            r_ = np.arcsinh(yv) - np.arcsinh(out["q90_raw"])
            o = np.argsort(r_)
            cw = np.cumsum(gw_va[o])
            mdl.conf_shift = float(r_[o][np.searchsorted(cw, 0.9 * cw[-1])])
            out = mdl.outputs(X[va])
            q90c = np.sinh(np.arcsinh(out["q90_raw"]) + mdl.conf_shift)
            return out, float((yv <= q90c).mean()), float(lam)

        # A1: no monotone constraint
        out1, cov1, lam1 = run_flow(mono=False)
        A["A1_no_monotone"] = score(out1, S9_of(out1["S"]), out1["viol"], cov1)
        A["A1_no_monotone"]["lambda"] = lam1
        log(f"{ax}: A1 done (viol {out1['viol']:.4%}, twIBS "
            f"{A['A1_no_monotone']['twIBS']})")

        # A3: lambda=0 (frozen F; isotonic calibrator and conformal shift re-fitted)
        Xp_v, lab_v, _, _ = mc.make_pairs(X[va][:, M.xa_cols],
                                          np.minimum(yv, t_max), t_max,
                                          t1, t2, w3, keys_va)
        zf = M.F.predict(Xp_v, raw_score=True)
        gw_p = np.repeat(gw_va, mc.K)
        iso0 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso0.fit(mc.sigmoid(zf), lab_v, sample_weight=gw_p)
        m3 = mc.MeldLoc(ax, M.F, M.G, 0.0, iso0, 0.0, t_max, tg, M.fam_w,
                        M.xa_cols, M.xs_cols, meta={"ablation": "A3"})
        out3 = m3.outputs(X[va])
        r_ = np.arcsinh(yv) - np.arcsinh(out3["q90_raw"])
        o = np.argsort(r_)
        cw = np.cumsum(gw_va[o])
        m3.conf_shift = float(r_[o][np.searchsorted(cw, 0.9 * cw[-1])])
        q90c3 = np.sinh(np.arcsinh(out3["q90_raw"]) + m3.conf_shift)
        A["A3_lambda0_TPfree"] = score(out3, S9_of(out3["S"]), out3["viol"],
                                       float((yv <= q90c3).mean()))
        log(f"{ax}: A3 done")

        # A4: no family re-weighting
        out4, cov4, lam4 = run_flow(reweight=False)
        A["A4_no_reweight"] = score(out4, S9_of(out4["S"]), out4["viol"], cov4)
        A["A4_no_reweight"]["lambda"] = lam4
        log(f"{ax}: A4 done")

        # A7: latest values only (last value of each signal in the 2 s window, 5-d)
        out7, cov7, lam7 = run_flow(feat="last")
        A["A7_last_only"] = score(out7, S9_of(out7["S"]), out7["viol"], cov7)
        A["A7_last_only"]["lambda"] = lam7
        log(f"{ax}: A7 (latest values only) done")

        # A5: rank by P(Y>T1) instead of R (no re-training)
        allg = np.arange(int(gm.sum()))
        p1g = ref["p1"][gm]
        A["A5_pT1_ranking"] = {
            "mid_macro_pT1": round(macro(p1g, allg), 4),
            "mid_macro_R": round(macro(ref["R"][gm], allg), 4),
            "mid_pooled_pT1": round(mc.spearman(ref["p1"][mid], yv[mid]), 4),
            "mid_pooled_R": round(mc.spearman(ref["R"][mid], yv[mid]), 4),
            "pT1_vs_R_CI": [round(v, 4) for v in
                            mc.paired_group_ci(macro, p1g, ref["R"][gm],
                                               cv[gm], N_BOOT)],
            "families": {f: {
                "R": round(mc.spearman(ref["R"][gm][fam_gm == f],
                                       yv[gm][fam_gm == f]), 3),
                "pT1": round(mc.spearman(p1g[fam_gm == f],
                                         yv[gm][fam_gm == f]), 3),
                "n_anchors": int((fam_gm == f).sum()),
                "n_groups": int(len(set(cv[gm][fam_gm == f])))} for f in ok_f}}
        log(f"{ax}: A5 done")
        res["axes"][ax] = A

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
