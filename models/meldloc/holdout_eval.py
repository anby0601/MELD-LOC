#!/usr/bin/env python3
"""One-shot in-route holdout evaluation (24 unseen runs on routeA).

Procedure: A) re-verify the 12 items of the freeze manifest (abort on any mismatch)
      B) extract features/labels of the holdout runs with the same pipeline as the cache
         (written to a separate npz)
      C) score the registered metrics: all runs / dev-lineage recipes / train-lineage recipes.
         Performed once; no re-evaluation or re-design afterwards.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "models/features"))
sys.path.insert(0, str(HERE / "models/meldloc"))
sys.path.insert(0, str(HERE / "models/baselines"))
import build_features as hc  # noqa: E402
import meldloc_core as mc  # noqa: E402
from distributional_baselines import qmulti_curve  # noqa: E402

MODEL_DIR = HERE / "models/frozen"
MANIFEST = HERE / "results/evaluation/model_freeze_manifest.json"
LEDGER = HERE / "data/logs/holdout/ledger.csv"
CACHE = HERE / "results/features/development_features.npz"
OUT = HERE / "results/evaluation/holdout_eval.json"
HO_NPZ = HERE / "results/features/holdout_features.npz"
N_BOOT = 600

META = {  # sid -> (family, group, lineage)
 "ho_ls08": ("lateral_shear", "nonrigid|lateral_shear|0.8", "dev"),
 "ho_rd_gradsweep": ("random_dropout", "legacy|dropout_grad_sweep", "dev"),
 "ho_rd_7090": ("random_dropout", "exp|dropout_uniform_grad_70_90", "dev"),
 "ho_rd_6080": ("random_dropout", "exp|dropout_uniform_grad_60_80", "dev"),
 "ho_rd_temporal80": ("random_dropout", "legacy|dropout_temporal_80", "train"),
 "ho_rd_uniform60": ("random_dropout", "legacy|dropout_uniform_sweep", "train"),
 "ho_rc_early12": ("range_clip", "legacy|range_early_12", "dev"),
 "ho_sr_right125": ("sector_remove", "legacy|sector_right_sweep_125_140", "train"),
 "ho_sr_rear120": ("sector_remove", "legacy|sector_rear_sweep", "train"),
 "ho_sr_right140": ("sector_remove", "exp|sector_right_140", "train"),
 "ho_ys15": ("yaw_shear", "nonrigid|yaw_shear|1.5", "dev"),
 "ho_yr_1p2": ("yaw_rotate", "exp|yaw_rotate_1_2deg", "dev"),
}
H16 = lambda b: hashlib.sha256(b).hexdigest()[:16]


def verify_freeze():
    man = json.loads(MANIFEST.read_text())
    fails = []
    if mc.version_hash() != man["code_version_hash_now"]:
        fails.append("code version_hash")
    if H16(json.dumps(hc.LON_OFFSETS, sort_keys=True).encode()) != man["shared"]["LON_OFFSETS_sha"]:
        fails.append("LON_OFFSETS")
    if H16(json.dumps([mc.T1, mc.T2, mc.W3], sort_keys=True).encode()) != man["shared"]["T1_T2_W3_sha"]:
        fails.append("T1/T2/W3")
    if mc.N_RGRID != man["shared"]["N_RGRID"] or mc.N_TGRID != man["shared"]["N_TGRID"]:
        fails.append("grid constants")
    models = {}
    for ax in ("lon", "lat", "yaw"):
        p = MODEL_DIR / f"meldloc_{ax}.joblib"
        a = man["axes"][ax]
        if hashlib.sha256(p.read_bytes()).hexdigest()[:16] != a["file_sha256"]:
            fails.append(f"{ax} file")
        m = mc.MeldLoc.load(p)
        iso_sig = np.concatenate([np.asarray(m.iso.X_thresholds_),
                                  np.asarray(m.iso.y_thresholds_)])
        checks = [
            (H16(m.F.booster_.model_to_string().encode()), a["F_sha"], "F"),
            (H16(m.G.booster_.model_to_string().encode()), a["G_sha"], "G"),
            (m.lam, a["lambda"], "lambda"),
            (H16(iso_sig.tobytes()), a["h_isotonic_sha"], "h"),
            (round(m.conf_shift, 12), a["conformal_shift"], "shift"),
            (round(m.t_max, 9), a["t_max"], "t_max"),
            (H16(np.asarray(m.t_grid).tobytes()), a["t_grid_sha"], "t_grid"),
            (H16(np.linspace(mc.T1[ax], mc.W3[ax], mc.N_RGRID).tobytes()),
             a["R_grid_sha"], "R_grid"),
            (H16(json.dumps({k: round(v, 10) for k, v in m.fam_w.items()},
                            sort_keys=True).encode()), a["family_weights_sha"], "famW"),
            (H16(np.asarray(m.xa_cols).tobytes() + np.asarray(m.xs_cols).tobytes()),
             a["feature_masks_sha"], "masks"),
        ]
        for got, want, nm in checks:
            if got != want:
                fails.append(f"{ax} {nm}")
        models[ax] = m
    return models, fails


def extract_holdout():
    """Extract holdout features/labels with the same pipeline as the cache (separate npz)."""
    rows = list(csv.reader(open(LEDGER)))[1:]
    ok = {}
    for r in rows:
        if r[3] == "OK" and r[4] == "1":
            ok[(r[0], r[1])] = r[6]
    X, Y, FAM, GRP, MEM, SCEN, TA = [], [], [], [], [], [], []
    for (sid, rep), d in sorted(ok.items()):
        fam, grp, mem = META[sid]
        dd, why = hc.load_run(Path(d) if Path(d).is_absolute() else HERE / d)
        assert dd is not None, f"{sid} load fail: {why}"
        t = dd["timestamp"]
        hi = np.searchsorted(t, t + hc.H, side="left")
        run_off = hc.LON_OFFSETS["routeA"]
        lab = {}
        for ax in hc.AX:
            e = np.abs(dd[hc.COL[ax]] - (run_off if ax == "lon" else 0.0))
            lab[ax] = np.array([e[i:max(j, i + 1)].max() for i, j in enumerate(hi)])
        anchors = np.arange(t[0] + hc.HIST_MIN, t[-1] - hc.H, hc.STRIDE)
        for tau in anchors:
            i = np.searchsorted(t, tau, side="right") - 1
            if i < 0 or tau - t[i] > 0.5:
                continue
            f, bad = [], False
            for L in hc.SCALES:
                for s in hc.SIG:
                    w = hc.window_stats(t, dd[s], tau - L, tau)
                    if w is None:
                        bad = True
                        break
                    f += w
                if bad:
                    break
            if bad:
                continue
            X.append(f)
            Y.append([lab[ax][i] for ax in hc.AX])
            FAM.append(fam); GRP.append(grp); MEM.append(mem)
            SCEN.append(f"{sid}_r{rep}"); TA.append(tau)
    return (np.array(X, np.float64), np.array(Y, np.float64), np.array(FAM),
            np.array(GRP), np.array(MEM), np.array(SCEN), np.array(TA, np.float64))


def main():
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    models, fails = verify_freeze()
    if fails:
        print("FREEZE_MISMATCH - evaluation aborted:", fails)
        sys.exit(3)
    log("freeze manifest: all 12 items match")

    Xh, Yh, FAM, GRP, MEM, SCEN, TA = extract_holdout()
    np.savez_compressed(HO_NPZ, X=Xh, Y=Yh, family=FAM, group=GRP, member=MEM,
                        scenario=SCEN, t_anchor=TA)
    log(f"holdout extraction: {len(set(SCEN))} runs, {len(Xh):,} anchors -> {HO_NPZ.name}")

    Z = np.load(CACHE, allow_pickle=True)
    Xc = Z["X"].astype(np.float64); Yc = Z["Y"].astype(np.float64)
    AXN = list(Z["axes"].astype(str)); split = Z["split"].astype(str)
    tr = split == "train"
    fname = list(Z["fname"].astype(str))
    tp_cols = np.array([f.startswith("transform_probability|") for f in fname])
    gbdt = mc.GBDT_DEFAULT
    CONTROL = {"translate_y", "yaw_rotate", "translate_x"}

    res = {"status": "HOLDOUT_ONE_SHOT",
           "freeze_verified": True, "n_runs": len(set(SCEN)),
           "n_anchors": int(len(Xh)), "axes": {}}
    subsets = {"all": np.ones(len(Xh), bool),
               "dev_lineage": MEM == "dev", "train_lineage": MEM == "train"}

    for ax in ("lon", "lat", "yaw"):
        ai = AXN.index(ax)
        y = Yc[:, ai]; yh = Yh[:, ai]
        t1, t2, w3 = mc.T1[ax], mc.T2[ax], mc.W3[ax]
        M = models[ax]
        out = M.outputs(Xh)
        S, tg = out["S"], M.t_grid
        at = float(np.arcsinh(M.t_max))
        gw = mc.group_equal_weights(GRP)

        # baselines (deterministic re-fit on the frozen training split = the frozen baselines)
        P = np.sinh(lgb.LGBMRegressor(objective="l1", **gbdt)
                    .fit(Xc[tr], np.arcsinh(y[tr])).predict(Xh))
        q50b = np.sinh(lgb.LGBMRegressor(objective="quantile", alpha=0.5, **gbdt)
                       .fit(Xc[tr], np.arcsinh(y[tr])).predict(Xh))
        q90b = np.sinh(lgb.LGBMRegressor(objective="quantile", alpha=0.9, **gbdt)
                       .fit(Xc[tr], np.arcsinh(y[tr])).predict(Xh))
        C2 = lgb.LGBMClassifier(is_unbalance=True, **gbdt).fit(
            Xc[tr], (y[tr] > t1).astype(int)).predict_proba(Xh)[:, 1]
        TPC = lgb.LGBMClassifier(is_unbalance=True, **gbdt).fit(
            Xc[tr][:, tp_cols], (y[tr] > t1).astype(int)
        ).predict_proba(Xh[:, tp_cols])[:, 1]
        sc = StandardScaler().fit(Xc[tr])
        LOGR = LogisticRegression(max_iter=2000).fit(
            sc.transform(Xc[tr]), (y[tr] > t1).astype(int)
        ).predict_proba(sc.transform(Xh))[:, 1]
        tg9 = np.sinh(np.linspace(0, at, 9))
        S_ind = np.stack([lgb.LGBMClassifier(is_unbalance=True, **gbdt).fit(
            Xc[tr], (y[tr] > tv).astype(int)).predict_proba(Xh)[:, 1]
            for tv in tg9], axis=1)
        AQ = np.stack([lgb.LGBMRegressor(objective="quantile", alpha=q, **gbdt
                       ).fit(Xc[tr], np.arcsinh(y[tr])).predict(Xh)
                       for q in (.1, .2, .3, .4, .5, .6, .7, .8, .9)], axis=1)
        Sq = qmulti_curve(np.maximum.accumulate(np.maximum(AQ, 0.0), axis=1),
                          np.arcsinh(tg9))
        S9 = np.stack([mc.prob_at(S, tg, tv) for tv in tg9], axis=1)
        log(f"{ax}: outputs and baselines ready (viol={out['viol']:.4%})")

        band = np.where(yh <= t1, "normal", np.where(yh <= w3, "middle", "div"))
        w_t = np.where((tg9 >= t1) & (tg9 <= w3), 2.0, 1.0)
        A = {"monotone_violation_pre_projection": round(out["viol"], 6)}

        for sub, msub in subsets.items():
            if not msub.any():
                continue
            ys, q90c = yh[msub], out["q90_conf"][msub]
            lab_t = (ys[:, None] > tg9[None, :]).astype(float)
            ib = lambda Sm: float((w_t[None, :] * (Sm[msub] - lab_t) ** 2).mean())
            gsub = GRP[msub]
            gws = mc.group_equal_weights(gsub)
            p1 = out["p_T1"][msub]
            lab1 = (ys > t1).astype(float)
            # ECE (10-bin, group-weighted), the registered definition: threshold-augmented
            # K=10 pairs, calibrated probability vs label (training step 7 repeated on the holdout)
            keys = mc.anchor_keys(SCEN[msub], TA[msub])
            ycap_h = np.minimum(ys, M.t_max)
            Xp_h, lab_p, _, _ = mc.make_pairs(Xh[msub][:, M.xa_cols], ycap_h,
                                              M.t_max, t1, t2, w3, keys)
            Xs_h = np.hstack([np.repeat(Xh[msub][:, M.xs_cols], mc.K, axis=0),
                              Xp_h[:, -1:]]).astype(np.float32)
            zf = M.F.predict(Xp_h, raw_score=True)
            zg = M.G.predict(Xs_h, raw_score=True)
            p_pair = M.iso.predict(mc.sigmoid(zf + M.lam * zg))
            gwp = np.repeat(gws, mc.K)
            bins = np.clip((p_pair * 10).astype(int), 0, 9)
            ece = 0.0
            for b in range(10):
                mb = bins == b
                if mb.sum() < 20:
                    continue
                ece += gwp[mb].sum() / gwp.sum() * abs(
                    float(np.average(lab_p[mb], weights=gwp[mb])) -
                    float(np.average(p_pair[mb], weights=gwp[mb])))
            mid = msub & (band == "middle") & ~np.isin(FAM, list(CONTROL))
            fams_mid = FAM[mid]
            ok_f = [f for f in sorted(set(fams_mid)) if (fams_mid == f).sum() >= 20]
            def macro(s):
                vals = [mc.spearman(s[FAM[mid] == f], yh[mid][FAM[mid] == f])
                        for f in ok_f]
                vals = [v for v in vals if np.isfinite(v)]
                return float(np.mean(vals)) if vals else float("nan")
            e_med = out["e_med"][msub]
            o = np.argsort(-e_med, kind="mergesort")
            k = int(len(o) * 0.1)
            lab2, lab3 = (ys > t2).astype(int), (ys > w3).astype(int)
            A[sub] = {
                "n_anchors": int(msub.sum()),
                "q90_coverage": {"MELD_conf": round(float((ys <= q90c).mean()), 4),
                                 "Q_gbdt": round(float((ys <= q90b[msub]).mean()), 4),
                                 "MELD_mean_asinh_width": round(float(np.arcsinh(q90c).mean()), 3)},
                "cov_by_band": {b: round(float((ys[band[msub] == b] <=
                                                q90c[band[msub] == b]).mean()), 3)
                                for b in ("normal", "middle", "div")
                                if (band[msub] == b).sum() >= 20},
                "ECE_pT1": round(ece, 4),
                "twIBS": {"MELD": round(ib(S9), 5),
                          "K_indep": round(ib(S_ind), 5),
                          "Q_multi": round(ib(Sq), 5)},
                "mid_n_by_family": {f: int((fams_mid == f).sum())
                                    for f in sorted(set(fams_mid))},
                "mid_macro_R": round(macro(out["R"][msub][ (band[msub]=="middle") & ~np.isin(FAM[msub], list(CONTROL)) ] if False else out["R"][mid]), 4),
                "mid_macro_P": round(macro(P[mid]), 4),
                "auroc_T1": {"MELD": round(mc.auroc(p1, lab1.astype(int)), 5),
                             "C2": round(mc.auroc(C2[msub], lab1.astype(int)), 5),
                             "TP_GBDT": round(mc.auroc(TPC[msub], lab1.astype(int)), 5),
                             "logistic": round(mc.auroc(LOGR[msub], lab1.astype(int)), 5)},
                "auroc_T2_T3W": {"MELD": [round(mc.auroc(out["p_T2"][msub], lab2), 5),
                                          round(mc.auroc(out["p_3W"][msub], lab3), 5)]},
                "top10_retained": {"MELD": round(float(ys[o][k:].mean()), 4),
                                   "P_asinh": round(float(ys[np.argsort(-P[msub], kind='mergesort')][k:].mean()), 4)},
                "mae": {"MELD": round(float(np.abs(e_med - ys).mean()), 4),
                        "q50": round(float(np.abs(q50b[msub] - ys).mean()), 4)},
            }
            log(f"{ax}/{sub}: cov {A[sub]['q90_coverage']['MELD_conf']} "
                f"ECE {A[sub]['ECE_pT1']} twIBS M{A[sub]['twIBS']['MELD']} "
                f"Q{A[sub]['twIBS']['Q_multi']}")
        res["axes"][ax] = A

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
