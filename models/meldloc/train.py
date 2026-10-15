#!/usr/bin/env python3
"""MELD-Loc training: the registered single-pass flow (order fixed).

  1. unweighted core F -> cluster-wise OOF logits
  2. one family re-weighting from the unweighted OOF loss (cap [0.5, 2]) -> fixed
  3. weighted F on the training split -> weighted cluster-wise OOF logits
  4. residual adapter G (weighted) with those OOF logits as init_score
  5. lambda chosen once on dev-val ({0, .25, .5, 1}, weighted final models)
  6. final model
  7. probability calibration: isotonic h on the natural dev-val distribution (equal group weights)
  8. conformal-style 90% one-sided upper bound: dev-val group residual quantile (additive in asinh space)
  9. frozen serialization (F, G, lambda, h, shift, t_max, grids, feature ids, family weights, version hash)

The training report records the cohort hash.
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
MODEL_DIR = HERE / "models/trained"       # the shipped models stay in models/frozen
REPORT = HERE / "models/trained/train_report.json"


def main():
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default="lon,lat,yaw")
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--gbdt-json", default="", help="GBDT hyper-parameter override (JSON)")
    a = ap.parse_args()
    gbdt = dict(mc.GBDT_DEFAULT)
    if a.gbdt_json:
        gbdt.update(json.loads(a.gbdt_json))

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
    t_anchor = Z["t_anchor"].astype(np.float64)
    tr, va = split == "train", split == "validation"

    tp_cols = np.array([f.startswith("transform_probability|") for f in fname])
    xa_cols = ~tp_cols
    keys_tr = mc.anchor_keys(scen[tr], t_anchor[tr])
    keys_va = mc.anchor_keys(scen[va], t_anchor[va])
    fold_tr, _ = mc.cluster_folds(clus[tr])
    gw_va = mc.group_equal_weights(grp[va])   # calibration/conformal unit = recipe group
    log(f"train {tr.sum():,} / dev-val {va.sum():,} | TP-free {int(xa_cols.sum())}d "
        f"TP {int(tp_cols.sum())}d | version {mc.version_hash()}")

    report = {"version_hash": mc.version_hash(), "feature_set": mc.FEATURE_SET_ID,
              "gbdt": gbdt, "cache": str(a.cache),
              "status": "TRAINING (the models every result refers to are the shipped ones in models/frozen)",
              "deviations": ["conformal/calibration unit = recipe_group (cluster_id fallback for indices that predate recipe groups)"],
              "axes": {}}

    for ax in (x.strip() for x in a.axes.split(",")):
        ai = AXN.index(ax)
        y = Y[:, ai]
        t1, t2, w3 = mc.T1[ax], mc.T2[ax], mc.W3[ax]
        t_max = float(np.quantile(y[tr], mc.TMAX_Q))
        at = float(np.arcsinh(t_max))
        ycap = np.minimum(y, t_max)
        t_grid = np.sinh(np.linspace(0.0, at, mc.N_TGRID))

        Xp, lab, w0, aidx = mc.make_pairs(X[tr][:, xa_cols], ycap[tr], t_max,
                                          t1, t2, w3, keys_tr)
        fold_p = fold_tr[aidx]
        fam_p = fam[tr][aidx]
        mono_a = [0] * int(xa_cols.sum()) + [-1]
        mono_s = [0] * int(tp_cols.sum()) + [-1]

        def fit_F(w, fit_mask=None):
            m = lgb.LGBMClassifier(monotone_constraints=mono_a,
                                   monotone_constraints_method="advanced", **gbdt)
            s = slice(None) if fit_mask is None else fit_mask
            m.fit(Xp[s], lab[s], sample_weight=w[s])
            return m

        # ---- 1. unweighted F + OOF ----
        oof_u = np.zeros(len(lab))
        for k in range(5):
            m = fit_F(w0, fold_p != k)
            oof_u[fold_p == k] = m.predict_proba(Xp[fold_p == k])[:, 1]
        # ---- 2. family re-weighting (once) ----
        wf = mc.family_weights(oof_u, lab, fam_p)
        w_rw = w0 * np.array([wf[f] for f in fam_p])
        log(f"{ax}: family weights {json.dumps({k: round(v, 2) for k, v in wf.items()})}")
        # ---- 3. weighted F + OOF ----
        oof_w = np.zeros(len(lab))
        for k in range(5):
            m = fit_F(w_rw, fold_p != k)
            oof_w[fold_p == k] = m.predict_proba(Xp[fold_p == k])[:, 1]
        F_full = fit_F(w_rw)
        # ---- 4. residual adapter G (init_score = weighted OOF logits) ----
        z_oof = mc.logit(oof_w)
        Xs_p = np.hstack([np.repeat(X[tr][:, tp_cols], mc.K, axis=0),
                          Xp[:, -1:]]).astype(np.float32)
        G = lgb.LGBMClassifier(monotone_constraints=mono_s,
                               monotone_constraints_method="advanced", **gbdt)
        G.fit(Xs_p, lab, sample_weight=w_rw, init_score=z_oof)

        model = mc.MeldLoc(ax, F_full, G, 1.0, None, 0.0, t_max, t_grid, wf,
                           xa_cols, tp_cols,
                           meta={"trained_at": time.strftime("%F %T"),
                                 "version": mc.version_hash(), "gbdt": gbdt})

        # ---- 5. lambda selection (once, on dev-val) ----
        # K=10 anchor-deterministic threshold samples on dev-val, compared by weighted log-loss
        ycap_v = np.minimum(y[va], t_max)
        Xp_v, lab_v, w0_v, aidx_v = mc.make_pairs(X[va][:, xa_cols], ycap_v, t_max,
                                                  t1, t2, w3, keys_va)
        Xs_v = np.hstack([np.repeat(X[va][:, tp_cols], mc.K, axis=0),
                          Xp_v[:, -1:]]).astype(np.float32)
        zf_v = F_full.predict(Xp_v, raw_score=True)
        zg_v = G.predict(Xs_v, raw_score=True)
        gw_pairs = np.repeat(gw_va, mc.K)
        eps = 1e-7
        lam_ll = {}
        for lam in mc.LAM_GRID:
            p = mc.sigmoid(zf_v + lam * zg_v)
            ll = -(lab_v * np.log(p + eps) + (1 - lab_v) * np.log(1 - p + eps))
            lam_ll[lam] = float((ll * gw_pairs).sum() / gw_pairs.sum())
        best_lam = min(lam_ll, key=lam_ll.get)
        model.lam = float(best_lam)
        log(f"{ax}: lambda={best_lam} (dev-val group-weighted log-loss "
            f"{json.dumps({str(k): round(v, 5) for k, v in lam_ll.items()})})")

        # ---- 6-7. final model + isotonic calibrator h (dev-val, equal group weights) ----
        p_v = mc.sigmoid(zf_v + best_lam * zg_v)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_v, lab_v, sample_weight=gw_pairs)
        model.iso = iso
        # calibration error before/after (10-bin ECE, group-weighted)
        def ece(p):
            bins = np.clip((p * 10).astype(int), 0, 9)
            e = 0.0
            for b in range(10):
                m = bins == b
                if m.sum() < 20:
                    continue
                wsum = gw_pairs[m].sum()
                e += wsum / gw_pairs.sum() * abs(
                    float(np.average(lab_v[m], weights=gw_pairs[m])) -
                    float(np.average(p[m], weights=gw_pairs[m])))
            return e
        ece_raw, ece_cal = ece(p_v), ece(iso.predict(p_v))

        # ---- 8. conformal-style 90% one-sided upper bound (dev-val group residual quantile) ----
        _, viol = model.curve(X[va], calibrated=True, return_violation=True)
        out_v = model.outputs(X[va])
        r = np.arcsinh(y[va]) - np.arcsinh(out_v["q90_raw"])
        # group-weighted 90th percentile -> additive shift in asinh space
        o = np.argsort(r)
        cw = np.cumsum(gw_va[o])
        shift = float(r[o][np.searchsorted(cw, 0.9 * cw[-1])])
        # the shift is signed: if the raw quantile over-covers (cov > 0.9) a negative shift tightens it
        # (the target is P(Y <= q0.9) ~ 0.9, not "widen only")
        model.conf_shift = shift
        cov_raw = float((y[va] <= out_v["q90_raw"]).mean())
        q90c = np.sinh(np.arcsinh(out_v["q90_raw"]) + model.conf_shift)
        cov_conf = float((y[va] <= q90c).mean())
        log(f"{ax}: ECE {ece_raw:.4f}->{ece_cal:.4f} | monotonicity violations (before projection) {viol:.4%} "
            f"| q90 cov {cov_raw:.4f}→{cov_conf:.4f} (shift {model.conf_shift:.4f})")

        # ---- 9. serialization ----
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"meldloc_{ax}.joblib"
        model.save(path)
        report["axes"][ax] = {
            "t_max": t_max, "lambda": model.lam, "family_weights": wf,
            "lam_grid_logloss": {str(k): v for k, v in lam_ll.items()},
            "ece_raw": ece_raw, "ece_calibrated": ece_cal,
            "monotone_violation_pre_projection": viol,
            "conformal_shift_asinh": model.conf_shift,
            "q90_coverage_raw": cov_raw, "q90_coverage_conformal": cov_conf,
            "model_path": str(path)}
        log(f"{ax}: saved -> {path}")

    REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    log(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
