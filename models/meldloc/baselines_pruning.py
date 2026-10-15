#!/usr/bin/env python3
"""Literature baselines and the grouped feature-pruning ablation on the same data and splits.

  survival   : survival stacking [Craig 2021]: discrete-hazard labels on the same 41-point asinh grid,
               S(t_j) = prod_{k<=j} (1-h_k); same GBDT hyper-parameters and seed 20260803;
               twIBS, AUROC@T1, ECE@T1 and 90% bound coverage with the same conformal procedure fitted on dev-val.
  rearranged : rearrangement of Q-multi (9 quantile GBDTs) [Chernozhukov 2010]: (a) cumulative maximum (as in Q-multi),
               (b) sorting (the original rearrangement); crossing rate before projection, twIBS, coverage.
  featprune  : grouped feature pruning: dedup (5 duplicated last@8s + 10 ptp columns removed) and each of the
               12 statistic groups removed in turn; re-trained with the registered pipeline
               (seed_variability.train_axis, seed 20260803); dev-val and holdout metrics.

The frozen models are not read (baselines and ablations are all re-fitted). Output: the functional sub-folders of results/ (see models/README.md).
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
import seed_variability as sv  # noqa: E402

CACHE = HERE / "results/features/development_features.npz"
HO_NPZ = HERE / "results/features/holdout_features.npz"
CR_NPZ = HERE / "results/features/crossroute_features.npz"
RES = HERE / "results"
AXES = ("lon", "lat", "yaw")
SEED = 20260803
STATS = ["mean", "std", "min", "max", "last", "slope", "q25", "q50", "q75", "ptp", "mad", "acf1"]


def load_all():
    Z = np.load(CACHE, allow_pickle=True)
    sets = {"holdout": np.load(HO_NPZ, allow_pickle=True), "crossroute": np.load(CR_NPZ, allow_pickle=True)}
    return Z, sets


def set_metrics(ax, S9, tg9, y, extra=None):
    """Shared distributional metrics: twIBS, AUROC@T1 (S(T1)), ECE@T1 (10-bin, unweighted)."""
    t1, w3 = mc.T1[ax], mc.W3[ax]
    w_t = np.where((tg9 >= t1) & (tg9 <= w3), 2.0, 1.0)
    lab_t = (y[:, None] > tg9[None, :]).astype(float)
    return {"twIBS": round(float((w_t[None, :] * (S9 - lab_t) ** 2).mean()), 5)}


def conformal_from_curve(S, tg, y_dev, gw_dev, S_dev):
    """Same procedure as MELD: q90 = first t with S <= 0.1 on dev-val; shift = group-weighted 90th percentile of the asinh residuals."""
    q_dev = mc.first_below(S_dev, tg, 0.10)
    r = np.arcsinh(y_dev) - np.arcsinh(q_dev)
    o = np.argsort(r); cw = np.cumsum(gw_dev[o])
    shift = float(r[o][np.searchsorted(cw, 0.9 * cw[-1])])
    q = mc.first_below(S, tg, 0.10)
    return np.sinh(np.arcsinh(q) + shift), shift

def ece_pairs_from_curve(S, tg, X, y, t_max, ax, SCEN, TA, GRP):
    """ECE as in holdout_eval: K=10 deterministic threshold pairs per anchor, 10 bins, equal recipe-group weights.
    Any curve S on grid tg is interpolated at the pair thresholds, so baselines are scored with the MELD definition."""
    t1, t2, w3 = mc.T1[ax], mc.T2[ax], mc.W3[ax]
    keys = mc.anchor_keys(SCEN, TA)
    Xp, lab_p, _, aidx = mc.make_pairs(X[:, :1], np.minimum(y, t_max), t_max, t1, t2, w3, keys)
    at = float(np.arcsinh(t_max)); t_pair = np.sinh(Xp[:, -1].astype(np.float64) * at)
    p = np.empty(len(t_pair))
    for i in range(len(S)):
        sl = slice(i * mc.K, (i + 1) * mc.K)
        p[sl] = np.interp(t_pair[sl], tg, S[i])
    gw = np.repeat(mc.group_equal_weights(GRP), mc.K)
    bins = np.clip((p * 10).astype(int), 0, 9); e = 0.0
    for b in range(10):
        m = bins == b
        if m.sum() >= 20:
            e += gw[m].sum() / gw.sum() * abs(float(np.average(lab_p[m], weights=gw[m])) - float(np.average(p[m], weights=gw[m])))
    return float(e)


def set_meta(Z, va, Zh=None):
    if Zh is None:
        grp = Z["recipe_group"].astype(str) if "recipe_group" in Z else Z["cluster"].astype(str)
        return Z["scenario"].astype(str)[va], Z["t_anchor"].astype(np.float64)[va], grp[va]
    return Zh["scenario"].astype(str), Zh["t_anchor"].astype(np.float64), Zh["group"].astype(str)


def cmd_survival(Z, sets, log):
    import lightgbm as lgb
    X = Z["X"].astype(np.float64); Y = Z["Y"].astype(np.float64); AXN = list(Z["axes"].astype(str))
    split = Z["split"].astype(str); tr, va = split == "train", split == "validation"
    grp = Z["recipe_group"].astype(str) if "recipe_group" in Z else Z["cluster"].astype(str)
    gw_va = mc.group_equal_weights(grp[va])
    gbdt = dict(mc.GBDT_DEFAULT); gbdt["random_state"] = SEED
    res = {"method": "survival stacking (discrete hazard on 41-pt asinh grid, GBDT classifier, no monotone constraint)",
           "seed": SEED, "axes": {}}
    for ax in AXES:
        ai = AXN.index(ax); y = Y[:, ai]; t1, t2, w3 = mc.T1[ax], mc.T2[ax], mc.W3[ax]
        t_max = float(np.quantile(y[tr], mc.TMAX_Q)); at = float(np.arcsinh(t_max))
        tg = np.sinh(np.linspace(0.0, at, mc.N_TGRID)); J = len(tg)
        ycap = np.minimum(y, t_max)
        # discrete event interval: j* = first j with Y_cap <= t_j  (t_0 = 0, so Y=0 gives j*=0)
        jstar = np.searchsorted(tg, ycap[tr], side="left")           # 0..J-1
        jstar = np.minimum(jstar, J - 1)
        # risk-set rows: j=0..j*  (label 1 at j*)
        n_rows = int((jstar + 1).sum())
        Xtr = X[tr].astype(np.float32); rows_x = np.empty((n_rows, Xtr.shape[1] + 1), np.float32)
        lab = np.zeros(n_rows, np.int32); pos = 0
        tf_grid = (np.arcsinh(tg) / at).astype(np.float32)
        for i in range(len(Xtr)):
            k = jstar[i] + 1
            rows_x[pos:pos + k, :-1] = Xtr[i]; rows_x[pos:pos + k, -1] = tf_grid[:k]
            lab[pos + k - 1] = 1; pos += k
        log(f"{ax}: hazard rows {n_rows:,} (mean risk-set len {n_rows/len(Xtr):.1f})")
        clf = lgb.LGBMClassifier(**gbdt).fit(rows_x, lab)

        def curve(Xe):
            Xe = Xe.astype(np.float32); S = np.empty((len(Xe), J), np.float32); surv = np.ones(len(Xe))
            for j in range(J):
                h = clf.predict_proba(np.hstack([Xe, np.full((len(Xe), 1), tf_grid[j], np.float32)]))[:, 1]
                surv = surv * (1.0 - h); S[:, j] = surv
            return S
        S_dev = curve(X[va]); y_dev = y[va]
        A = {"t_max": t_max, "sets": {}}
        tg9 = np.sinh(np.linspace(0, at, 9))
        targets = {"devval": (X[va], y_dev, S_dev, set_meta(Z, va))}
        for k, Zh in sets.items():
            Xe = Zh["X"].astype(np.float64); targets[k] = (Xe, Zh["Y"].astype(np.float64)[:, ai], curve(Xe), set_meta(Z, va, Zh))
        for k, (Xe, ye, S, (SC, TAe, GR)) in targets.items():
            S9 = np.stack([mc.prob_at(S, tg, tv) for tv in tg9], axis=1)
            q90c, shift = conformal_from_curve(S, tg, y_dev, gw_va, S_dev)
            band = np.where(ye <= t1, "normal", np.where(ye <= w3, "middle", "div"))
            p1 = mc.prob_at(S, tg, t1); lab1 = (ye > t1).astype(int)
            bins = np.clip((p1 * 10).astype(int), 0, 9); ece = 0.0
            for b in range(10):
                m = bins == b
                if m.sum() >= 20:
                    ece += m.mean() * abs(float(lab1[m].mean()) - float(p1[m].mean()))
            A["sets"][k] = {"n": int(len(ye)), "viol_raw": round(float((np.diff(S, axis=1) > 1e-6).mean()), 6),
                            **set_metrics(ax, S9, tg9, ye),
                            "auroc_T1": round(mc.auroc(p1, lab1), 5), "ECE_T1_unweighted": round(ece, 4),
                            "ECE_pairs": round(ece_pairs_from_curve(S, tg, Xe, ye, t_max, ax, SC, TAe, GR), 4),
                            "q90_coverage_conformal": round(float((ye <= q90c).mean()), 4),
                            "mean_width_asinh": round(float(np.arcsinh(q90c).mean()), 4),
                            "cov_by_band": {bb: round(float((ye[band == bb] <= q90c[band == bb]).mean()), 3)
                                            for bb in ("normal", "middle", "div") if (band == bb).sum() >= 20},
                            "conformal_shift": round(shift, 4)}
            log(f"{ax}/{k}: viol {A['sets'][k]['viol_raw']:.4%} twIBS {A['sets'][k]['twIBS']} AUROC {A['sets'][k]['auroc_T1']} "
                f"ECE {A['sets'][k]['ECE_T1_unweighted']} cov {A['sets'][k]['q90_coverage_conformal']} mid {A['sets'][k]['cov_by_band'].get('middle')}")
        res["axes"][ax] = A
        (RES / "baselines/survival_stacking.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("-> baseline_survival_stacking.json")


def cmd_rearranged(Z, sets, log):
    import lightgbm as lgb
    X = Z["X"].astype(np.float64); Y = Z["Y"].astype(np.float64); AXN = list(Z["axes"].astype(str))
    split = Z["split"].astype(str); tr, va = split == "train", split == "validation"
    grp = Z["recipe_group"].astype(str) if "recipe_group" in Z else Z["cluster"].astype(str)
    gw_va = mc.group_equal_weights(grp[va])
    gbdt = dict(mc.GBDT_DEFAULT); gbdt["random_state"] = SEED
    QL = (.1, .2, .3, .4, .5, .6, .7, .8, .9)
    res = {"method": "Q-multi (9 quantile GBDTs, asinh target) with (a) cummax [as used by Q-multi] vs (b) sorting rearrangement [Chernozhukov et al. 2010]",
           "seed": SEED, "axes": {}}
    for ax in AXES:
        ai = AXN.index(ax); y = Y[:, ai]; t1, w3 = mc.T1[ax], mc.W3[ax]
        t_max = float(np.quantile(y[tr], mc.TMAX_Q)); at = float(np.arcsinh(t_max))
        tg9 = np.sinh(np.linspace(0, at, 9)); atg = np.arcsinh(tg9)
        qrs = [lgb.LGBMRegressor(objective="quantile", alpha=q, **gbdt).fit(X[tr], np.arcsinh(y[tr])) for q in QL]
        A = {"sets": {}}
        targets = {"devval": (X[va], y[va], set_meta(Z, va))}
        targets.update({k: (Zh["X"].astype(np.float64), Zh["Y"].astype(np.float64)[:, ai], set_meta(Z, va, Zh)) for k, Zh in sets.items()})
        # dev-val curves (for the conformal shift)
        AQ_dev = np.stack([q.predict(X[va]) for q in qrs], axis=1)
        curves_dev = {"cummax": qmulti_curve(np.maximum.accumulate(np.maximum(AQ_dev, 0.0), axis=1), atg),
                      "sorted": qmulti_curve(np.sort(np.maximum(AQ_dev, 0.0), axis=1), atg),
                      "raw_unprojected": qmulti_curve(np.maximum(AQ_dev, 0.0), atg)}
        for k, (Xe, ye, (SC, TAe, GR)) in targets.items():
            AQ = np.stack([q.predict(Xe) for q in qrs], axis=1)
            cross = float((np.diff(AQ, axis=1) < -1e-6).mean())
            n_cross_anchor = float((np.diff(AQ, axis=1) < -1e-6).any(axis=1).mean())
            band = np.where(ye <= t1, "normal", np.where(ye <= w3, "middle", "div"))
            row = {"n": int(len(ye)), "quantile_crossing_rate_adjacent": round(cross, 4),
                   "anchors_with_any_crossing": round(n_cross_anchor, 4), "variants": {}}
            for name, proj in (("cummax", lambda a: np.maximum.accumulate(a, axis=1)),
                               ("sorted", lambda a: np.sort(a, axis=1)),
                               ("raw_unprojected", lambda a: a)):
                S = qmulti_curve(proj(np.maximum(AQ, 0.0)), atg)
                S_dev = curves_dev[name]
                q90c, shift = conformal_from_curve(S, tg9, y[va], gw_va, S_dev)
                p1 = mc.prob_at(S, tg9, t1); lab1 = (ye > t1).astype(int)
                row["variants"][name] = {"viol_curve_raw": round(float((np.diff(S, axis=1) > 1e-6).mean()), 6),
                                         **set_metrics(ax, S, tg9, ye),
                                         "auroc_T1": round(mc.auroc(p1, lab1), 5),
                                         "ECE_pairs": round(ece_pairs_from_curve(S, tg9, Xe, ye, t_max, ax, SC, TAe, GR), 4),
                                         "q90_coverage_conformal": round(float((ye <= q90c).mean()), 4),
                                         "mean_width_asinh": round(float(np.arcsinh(q90c).mean()), 4),
                                         "cov_mid": round(float((ye[band == "middle"] <= q90c[band == "middle"]).mean()), 3) if (band == "middle").sum() >= 20 else None,
                                         "conformal_shift": round(shift, 4)}
            A["sets"][k] = row
            log(f"{ax}/{k}: crossing {cross:.4f} | " + " ".join(f"{n}: twIBS {v['twIBS']} cov {v['q90_coverage_conformal']} mid {v['cov_mid']}"
                                                           for n, v in row["variants"].items()))
        res["axes"][ax] = A
        (RES / "baselines/rearranged_qmulti.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("-> baseline_rearranged_qmulti.json")


def subset_Z(Z, keep):
    d = {k: Z[k] for k in Z.files}
    d["X"] = Z["X"][:, keep]; d["fname"] = Z["fname"][keep]
    return d


def subset_set(Zh, keep):
    d = {k: Zh[k] for k in Zh.files}; d["X"] = Zh["X"][:, keep]; return d


def cmd_featprune(Z, sets, log, configs=None, n_jobs=12):
    fname = list(Z["fname"].astype(str))
    stat_of = np.array([f.split("|")[-1] for f in fname]); win_of = np.array([f.split("|")[1] for f in fname])
    all_cfg = {"full_120": np.ones(len(fname), bool),
               "dedup_105": ~(((stat_of == "last") & (win_of == "8s")) | (stat_of == "ptp"))}
    for st in STATS:
        all_cfg[f"drop_{st}"] = stat_of != st
    if configs:
        all_cfg = {k: v for k, v in all_cfg.items() if k in configs}
    gbdt = dict(mc.GBDT_DEFAULT); gbdt["random_state"] = SEED; gbdt["n_jobs"] = n_jobs
    path = RES / "ablations/feature_groups.json"
    res = json.loads(path.read_text()) if path.exists() else {"seed": SEED, "note": "registered pipeline retrained per config; frozen v1 untouched", "configs": {}}
    for name, keep in all_cfg.items():
        if name in res["configs"] and all(ax in res["configs"][name] for ax in AXES):
            log(f"{name}: already done"); continue
        Zs = subset_Z(Z, keep); Ss = {k: subset_set(v, keep) for k, v in sets.items()}
        res["configs"].setdefault(name, {"n_features": int(keep.sum())})
        for ax in AXES:
            if ax in res["configs"][name]:
                continue
            log(f"== {name} ({int(keep.sum())}d) / {ax}")
            M, devval = sv.train_axis(ax, Zs, SEED, gbdt, log)
            r = {"devval": {k: devval[k] for k in ("lambda", "ece_calibrated", "monotone_violation_pre_projection",
                                                   "conformal_shift_asinh", "q90_coverage_conformal", "cov_by_band")}}
            for k, Zh in Ss.items():
                r[k], _ = sv.eval_set(ax, M, Zh, log, k)
            res["configs"][name][ax] = r
            path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("-> feature_group_ablation.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmds", nargs="+", choices=["survival", "rearranged", "featprune"])
    ap.add_argument("--configs", default="", help="featprune: comma-separated subset of configurations")
    ap.add_argument("--n-jobs", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    for _d in ('ablations', 'baselines'): (RES / _d).mkdir(parents=True, exist_ok=True)
    Z, sets = load_all()
    for c in a.cmds:
        if c == "survival":
            cmd_survival(Z, sets, log)
        elif c == "rearranged":
            cmd_rearranged(Z, sets, log)
        else:
            cmd_featprune(Z, sets, log, [x for x in a.configs.split(",") if x] or None, a.n_jobs)


if __name__ == "__main__":
    main()
