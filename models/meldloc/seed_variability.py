#!/usr/bin/env python3
"""Seed-variability experiment: the registered training flow re-run with other seeds.

Replicates steps 1-8 of meldloc_train.py exactly, changing only (i) the GBDT random_state
(row/feature subsampling) and (ii) the cluster-fold assignment seed.
meldloc_core.py is not modified (its version hash is part of the freeze manifest).
The frozen models in models/frozen are neither read nor written; outputs go to models/seeds/.

Per seed: dev-val (lambda, ECE, shift, coverage) and the holdout/cross-route headline metrics
(monotonicity violations, coverage overall/by band, ECE@T1, twIBS, AUROC@T1, mid-band macro Spearman).
With --baselines the K-indep (9 threshold classifiers) and Q-multi (9 quantiles) baselines are
also re-fitted with the same seed (seed dependence of the baseline violation rates).
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
sys.path.insert(0, str(HERE / "models/baselines"))
import meldloc_core as mc  # noqa: E402
from distributional_baselines import qmulti_curve  # noqa: E402

CACHE = HERE / "results/features/development_features.npz"
HO_NPZ = HERE / "results/features/holdout_features.npz"
CR_NPZ = HERE / "results/features/crossroute_features.npz"
OUT_DIR = HERE / "results/seeds"
MODEL_ROOT = HERE / "models/seeds"
CONTROL = {"translate_y", "yaw_rotate", "translate_x"}


def ece10(p, lab, w):
    bins = np.clip((p * 10).astype(int), 0, 9)
    e = 0.0
    for b in range(10):
        m = bins == b
        if m.sum() < 20:
            continue
        e += w[m].sum() / w.sum() * abs(float(np.average(lab[m], weights=w[m])) -
                                        float(np.average(p[m], weights=w[m])))
    return float(e)


def train_axis(ax, Z, seed, gbdt, log):
    """Steps 1-8 of meldloc_train.py with a different seed."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    X = Z["X"].astype(np.float64); Y = Z["Y"].astype(np.float64)
    fname = list(Z["fname"].astype(str)); AXN = list(Z["axes"].astype(str))
    split = Z["split"].astype(str); fam = Z["family"].astype(str)
    clus = Z["cluster"].astype(str)
    grp = Z["recipe_group"].astype(str) if "recipe_group" in Z else clus
    scen = Z["scenario"].astype(str); t_anchor = Z["t_anchor"].astype(np.float64)
    tr, va = split == "train", split == "validation"
    tp_cols = np.array([f.startswith("transform_probability|") for f in fname])
    xa_cols = ~tp_cols
    keys_tr = mc.anchor_keys(scen[tr], t_anchor[tr])
    keys_va = mc.anchor_keys(scen[va], t_anchor[va])
    fold_tr, _ = mc.cluster_folds(clus[tr], seed=seed)          # fold seed
    gw_va = mc.group_equal_weights(grp[va])

    ai = AXN.index(ax); y = Y[:, ai]
    t1, t2, w3 = mc.T1[ax], mc.T2[ax], mc.W3[ax]
    t_max = float(np.quantile(y[tr], mc.TMAX_Q)); at = float(np.arcsinh(t_max))
    ycap = np.minimum(y, t_max)
    t_grid = np.sinh(np.linspace(0.0, at, mc.N_TGRID))
    Xp, lab, w0, aidx = mc.make_pairs(X[tr][:, xa_cols], ycap[tr], t_max, t1, t2, w3, keys_tr)
    fold_p = fold_tr[aidx]; fam_p = fam[tr][aidx]
    mono_a = [0] * int(xa_cols.sum()) + [-1]
    mono_s = [0] * int(tp_cols.sum()) + [-1]

    def fit_F(w, fit_mask=None):
        m = lgb.LGBMClassifier(monotone_constraints=mono_a,
                               monotone_constraints_method="advanced", **gbdt)
        s = slice(None) if fit_mask is None else fit_mask
        m.fit(Xp[s], lab[s], sample_weight=w[s])
        return m

    oof_u = np.zeros(len(lab))
    for k in range(5):
        m = fit_F(w0, fold_p != k)
        oof_u[fold_p == k] = m.predict_proba(Xp[fold_p == k])[:, 1]
    wf = mc.family_weights(oof_u, lab, fam_p)
    w_rw = w0 * np.array([wf[f] for f in fam_p])
    log(f"{ax}: family weights {json.dumps({k: round(v, 2) for k, v in wf.items()})}")
    oof_w = np.zeros(len(lab))
    for k in range(5):
        m = fit_F(w_rw, fold_p != k)
        oof_w[fold_p == k] = m.predict_proba(Xp[fold_p == k])[:, 1]
    F_full = fit_F(w_rw)
    z_oof = mc.logit(oof_w)
    Xs_p = np.hstack([np.repeat(X[tr][:, tp_cols], mc.K, axis=0), Xp[:, -1:]]).astype(np.float32)
    G = lgb.LGBMClassifier(monotone_constraints=mono_s,
                           monotone_constraints_method="advanced", **gbdt)
    G.fit(Xs_p, lab, sample_weight=w_rw, init_score=z_oof)
    model = mc.MeldLoc(ax, F_full, G, 1.0, None, 0.0, t_max, t_grid, wf, xa_cols, tp_cols,
                       meta={"trained_at": time.strftime("%F %T"), "version": mc.version_hash(),
                             "gbdt": gbdt, "seed": seed, "fold_seed": seed,
                             "purpose": "seed_variability"})

    ycap_v = np.minimum(y[va], t_max)
    Xp_v, lab_v, _, _ = mc.make_pairs(X[va][:, xa_cols], ycap_v, t_max, t1, t2, w3, keys_va)
    Xs_v = np.hstack([np.repeat(X[va][:, tp_cols], mc.K, axis=0), Xp_v[:, -1:]]).astype(np.float32)
    zf_v = F_full.predict(Xp_v, raw_score=True); zg_v = G.predict(Xs_v, raw_score=True)
    gw_pairs = np.repeat(gw_va, mc.K); eps = 1e-7
    lam_ll = {}
    for lam in mc.LAM_GRID:
        p = mc.sigmoid(zf_v + lam * zg_v)
        ll = -(lab_v * np.log(p + eps) + (1 - lab_v) * np.log(1 - p + eps))
        lam_ll[lam] = float((ll * gw_pairs).sum() / gw_pairs.sum())
    best_lam = min(lam_ll, key=lam_ll.get); model.lam = float(best_lam)
    p_v = mc.sigmoid(zf_v + best_lam * zg_v)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_v, lab_v, sample_weight=gw_pairs); model.iso = iso
    ece_raw, ece_cal = ece10(p_v, lab_v, gw_pairs), ece10(iso.predict(p_v), lab_v, gw_pairs)
    _, viol = model.curve(X[va], calibrated=True, return_violation=True)
    out_v = model.outputs(X[va])
    r = np.arcsinh(y[va]) - np.arcsinh(out_v["q90_raw"])
    o = np.argsort(r); cw = np.cumsum(gw_va[o])
    model.conf_shift = float(r[o][np.searchsorted(cw, 0.9 * cw[-1])])
    cov_raw = float((y[va] <= out_v["q90_raw"]).mean())
    q90c = np.sinh(np.arcsinh(out_v["q90_raw"]) + model.conf_shift)
    cov_conf = float((y[va] <= q90c).mean())
    band_v = np.where(y[va] <= t1, "normal", np.where(y[va] <= w3, "middle", "div"))
    devval = {"t_max": t_max, "lambda": model.lam, "family_weights": wf,
              "lam_grid_logloss": {str(k): v for k, v in lam_ll.items()},
              "ece_raw": ece_raw, "ece_calibrated": ece_cal,
              "monotone_violation_pre_projection": viol,
              "conformal_shift_asinh": model.conf_shift,
              "q90_coverage_raw": cov_raw, "q90_coverage_conformal": cov_conf,
              "cov_by_band": {b: round(float((y[va][band_v == b] <= q90c[band_v == b]).mean()), 4)
                              for b in ("normal", "middle", "div") if (band_v == b).sum() >= 20}}
    log(f"{ax}: λ={best_lam} ECE {ece_raw:.4f}→{ece_cal:.4f} viol {viol:.4%} "
        f"cov {cov_raw:.4f}→{cov_conf:.4f} shift {model.conf_shift:.4f}")
    return model, devval


def eval_set(ax, M, Zh, log, tag):
    """The 'all' subset metrics of holdout_eval.py (MELD only)."""
    Xh = Zh["X"].astype(np.float64); yh = Zh["Y"].astype(np.float64)[:, ("lon", "lat", "yaw").index(ax)]
    FAM = Zh["family"].astype(str); GRP = Zh["group"].astype(str)
    SCEN = Zh["scenario"].astype(str); TA = Zh["t_anchor"].astype(np.float64)
    t1, t2, w3 = mc.T1[ax], mc.T2[ax], mc.W3[ax]
    out = M.outputs(Xh); S, tg = out["S"], M.t_grid
    at = float(np.arcsinh(M.t_max)); tg9 = np.sinh(np.linspace(0, at, 9))
    S9 = np.stack([mc.prob_at(S, tg, tv) for tv in tg9], axis=1)
    band = np.where(yh <= t1, "normal", np.where(yh <= w3, "middle", "div"))
    w_t = np.where((tg9 >= t1) & (tg9 <= w3), 2.0, 1.0)
    lab_t = (yh[:, None] > tg9[None, :]).astype(float)
    twibs = lambda Sm: float((w_t[None, :] * (Sm - lab_t) ** 2).mean())
    q90c = out["q90_conf"]
    gws = mc.group_equal_weights(GRP)
    keys = mc.anchor_keys(SCEN, TA)
    Xp_h, lab_p, _, _ = mc.make_pairs(Xh[:, M.xa_cols], np.minimum(yh, M.t_max), M.t_max, t1, t2, w3, keys)
    Xs_h = np.hstack([np.repeat(Xh[:, M.xs_cols], mc.K, axis=0), Xp_h[:, -1:]]).astype(np.float32)
    p_pair = M.iso.predict(mc.sigmoid(M.F.predict(Xp_h, raw_score=True) +
                                      M.lam * M.G.predict(Xs_h, raw_score=True)))
    ece = ece10(p_pair, lab_p, np.repeat(gws, mc.K))
    mid = (band == "middle") & ~np.isin(FAM, list(CONTROL))
    fams_mid = FAM[mid]
    ok_f = [f for f in sorted(set(fams_mid)) if (fams_mid == f).sum() >= 20]
    vals = [mc.spearman(out["R"][mid][fams_mid == f], yh[mid][fams_mid == f]) for f in ok_f]
    vals = [v for v in vals if np.isfinite(v)]
    lab1 = (yh > t1).astype(int)
    res = {"n_anchors": int(len(yh)),
           "monotone_violation_pre_projection": round(float(out["viol"]), 6),
           "q90_coverage": round(float((yh <= q90c).mean()), 4),
           "cov_by_band": {b: round(float((yh[band == b] <= q90c[band == b]).mean()), 4)
                           for b in ("normal", "middle", "div") if (band == b).sum() >= 20},
           "ECE_pT1": round(ece, 4), "twIBS_MELD": round(twibs(S9), 5),
           "auroc_T1_MELD": round(mc.auroc(out["p_T1"], lab1), 5),
           "auroc_T2_MELD": round(mc.auroc(out["p_T2"], (yh > t2).astype(int)), 5),
           "auroc_3W_MELD": round(mc.auroc(out["p_3W"], (yh > w3).astype(int)), 5),
           "mid_macro_R": round(float(np.mean(vals)), 4) if vals else None,
           "mid_n": int(mid.sum()),
           "mean_asinh_width_q90": round(float(np.arcsinh(q90c).mean()), 4)}
    log(f"{ax}/{tag}: viol {res['monotone_violation_pre_projection']:.4%} cov {res['q90_coverage']} "
        f"mid {res['cov_by_band'].get('middle')} ECE {res['ECE_pT1']} twIBS {res['twIBS_MELD']} "
        f"AUROC {res['auroc_T1_MELD']}")
    return res, (S9, tg9, lab_t, w_t)


def baselines_axis(ax, Z, seed, gbdt, sets, log):
    """K-indep / Q-multi baselines with the same seed: violations and twIBS on dev-val, holdout and cross-route."""
    import lightgbm as lgb
    X = Z["X"].astype(np.float64); Y = Z["Y"].astype(np.float64)
    AXN = list(Z["axes"].astype(str)); split = Z["split"].astype(str)
    tr, va = split == "train", split == "validation"
    y = Y[:, AXN.index(ax)]
    t_max = float(np.quantile(y[tr], mc.TMAX_Q)); at = float(np.arcsinh(t_max))
    tg9 = np.sinh(np.linspace(0, at, 9))
    t1, w3 = mc.T1[ax], mc.W3[ax]
    w_t = np.where((tg9 >= t1) & (tg9 <= w3), 2.0, 1.0)
    clfs = [lgb.LGBMClassifier(is_unbalance=True, **gbdt).fit(X[tr], (y[tr] > tv).astype(int)) for tv in tg9]
    qrs = [lgb.LGBMRegressor(objective="quantile", alpha=q, **gbdt).fit(X[tr], np.arcsinh(y[tr]))
           for q in (.1, .2, .3, .4, .5, .6, .7, .8, .9)]
    out = {}
    targets = {"devval": (X[va], y[va])}
    targets.update({k: (Zh["X"].astype(np.float64), Zh["Y"].astype(np.float64)[:, AXN.index(ax)])
                    for k, Zh in sets.items()})
    for tag, (Xe, ye) in targets.items():
        S_ind = np.stack([c.predict_proba(Xe)[:, 1] for c in clfs], axis=1)
        AQ = np.stack([q.predict(Xe) for q in qrs], axis=1)
        viol_q = float((np.diff(AQ, axis=1) < -1e-6).mean())        # quantile crossing (before projection)
        Sq = qmulti_curve(np.maximum.accumulate(np.maximum(AQ, 0.0), axis=1), np.arcsinh(tg9))
        lab_t = (ye[:, None] > tg9[None, :]).astype(float)
        twibs = lambda Sm: float((w_t[None, :] * (Sm - lab_t) ** 2).mean())
        out[tag] = {"K_indep": {"viol": round(float((np.diff(S_ind, axis=1) > 1e-6).mean()), 4),
                                "twIBS": round(twibs(S_ind), 5)},
                    "Q_multi": {"viol_quantile_crossing": round(viol_q, 4),
                                "twIBS": round(twibs(Sq), 5)}}
        log(f"{ax}/{tag} baselines: K viol {out[tag]['K_indep']['viol']} twIBS {out[tag]['K_indep']['twIBS']} "
            f"| Q cross {viol_q:.4f} twIBS {out[tag]['Q_multi']['twIBS']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--axes", default="lon,lat,yaw")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="smoke test with n_estimators=20")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Z = np.load(CACHE, allow_pickle=True)
    sets = {"holdout": np.load(HO_NPZ, allow_pickle=True), "crossroute": np.load(CR_NPZ, allow_pickle=True)}
    for seed in (int(s) for s in a.seeds.split(",")):
        gbdt = dict(mc.GBDT_DEFAULT); gbdt["random_state"] = seed; gbdt["n_jobs"] = a.n_jobs
        if a.smoke:
            gbdt["n_estimators"] = 20
        tag = f"seed{seed}" + ("_smoke" if a.smoke else "")
        rep_path = OUT_DIR / f"{tag}.json"
        rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
        rep.update({"seed": seed, "fold_seed": seed, "gbdt": gbdt, "smoke": a.smoke,
                    "core_version_hash": mc.version_hash(), "frozen_v1_untouched": True})
        rep.setdefault("axes", {})
        for ax in (x.strip() for x in a.axes.split(",")):
            if ax in rep["axes"] and "holdout" in rep["axes"][ax] and not a.smoke:
                log(f"{tag}/{ax}: already done, skipped"); continue
            log(f"== {tag} / {ax} training")
            M, devval = train_axis(ax, Z, seed, gbdt, log)
            mp = MODEL_ROOT / tag / f"meldloc_{ax}.joblib"; M.save(mp)
            axrep = {"devval": devval, "model_path": str(mp.relative_to(HERE))}
            for k, Zh in sets.items():
                axrep[k], _ = eval_set(ax, M, Zh, log, k)
            if a.baselines:
                axrep["baselines"] = baselines_axis(ax, Z, seed, gbdt, sets, log)
            rep["axes"][ax] = axrep
            rep_path.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
            log(f"{tag}/{ax} -> {rep_path.name} ({(time.time()-t0)/60:.1f} min)")
    log("done")


if __name__ == "__main__":
    main()
