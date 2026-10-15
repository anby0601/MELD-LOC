#!/usr/bin/env python3
"""Result JSON -> LaTeX table fragments (reports/*.tex) and the full markdown tables.

Generates: condcov.tex, alarm_delay.tex, capraw.tex, seeds.tex, seeds_full.tex, timing.tex,
      litbase.tex, featprune.tex, dyncov.tex, stress.tex, stress_1col.tex (when the inputs exist).
Each fragment is a complete table environment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
RES = HERE / "results"
SV = RES / "seeds"
OUT = HERE / "reports"
AXES = ("lon", "lat", "yaw")


def f3(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


N_ANCH = {"holdout": 10827, "crossroute": 6023}


def cov3(v, n):
    """Coverage is stored with 4 decimals in the JSON; rounding again to 3 would double-round
    (0.907546 -> 0.9075 -> 0.907 instead of 0.908). Recover the integer hit count and round once."""
    if v is None:
        return "--"
    return f"{round(v * n) / n:.3f}"


def condcov():
    d = json.loads((RES / "coverage/conditional_coverage.json").read_text())
    L = [r"\begin{table}[H]",
         r"\caption{Conditional coverage of the $90\%$ bound binned on the predicted alarm probability $S_a(x,T_{1,a})$ (available online). For each bin: number of anchors, empirical exceedance rate of $T_{1,a}$, coverage with the frozen global shift $\delta_a$, and coverage with a per-bin (Mondrian) shift fitted on the development-validation split. Bins with fewer than 300 development-validation anchors retain the global shift.}",
         r"\label{tab:condcov}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.2pt}",
         r"\begin{tabular}{llrcccrccc}", r"\toprule",
         r" & & \multicolumn{4}{c}{Holdout (routeA)} & \multicolumn{4}{c}{Cross-route (routeB)} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"Axis & $S(T_1)$ bin & $n$ & exc. & cov. & cov.\,M & $n$ & exc. & cov. & cov.\,M \\", r"\midrule"]
    for ax in AXES:
        H = {b["bin"]: b for b in d["axes"][ax]["sets"]["holdout"]["bins"]}
        C = {b["bin"]: b for b in d["axes"][ax]["sets"]["crossroute"]["bins"]}
        bins = [b["bin"] for b in d["axes"][ax]["sets"]["holdout"]["bins"]]
        for i, b in enumerate(bins):
            h, c = H.get(b), C.get(b)
            lab = b.replace("[", "$[").replace(")", ")$")
            row = [ax if i == 0 else "", lab]
            for s in (h, c):
                if s is None:
                    row += ["--"] * 4
                else:
                    row += [str(s["n"]), f3(s["empirical_exceed_T1"], 2), f3(s["cov_frozen"]), f3(s["cov_mondrian"])]
            L.append(" & ".join(row) + r" \\")
        hs, cs = d["axes"][ax]["sets"]["holdout"], d["axes"][ax]["sets"]["crossroute"]
        L.append(f"{''} & marginal & {hs['n']} & -- & {cov3(hs['marginal_cov_frozen'], hs['n'])} & {cov3(hs['marginal_cov_mondrian'], hs['n'])} & "
                 f"{cs['n']} & -- & {cov3(cs['marginal_cov_frozen'], cs['n'])} & {cov3(cs['marginal_cov_mondrian'], cs['n'])} \\\\")
        if ax != "yaw":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "condcov.tex").write_text("\n".join(L) + "\n")


def alarm_delay(theta="0.5"):
    d = json.loads((RES / "timing/alarm_delay.json").read_text())
    L = [r"\begin{table}[!t]",
         r"\caption{Alarm latency on the final evaluation runs. An alarm is raised when the calibrated $S_a(x,T_{1,a}) \ge 0.5$ at a $0.5$\,s anchor. Sustained crossings are episodes in which the raw error stays above $T_{1,a}$ for at least the $1$\,s label window; transient crossings are any first exceedance. Delay is the first alarm time minus the crossing time (negative: the alarm precedes the crossing, which the $1$\,s look-ahead label permits). FA: false-alarm rate over anchors whose realized window error is below $T_{1,a}$.}",
         r"\label{tab:delay}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.4pt}",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Set & Axis & Sust.\ ep. & Det. & Median [s] & Max [s] & $\le 1$\,s & FA \\", r"\midrule"]
    for k, name in (("holdout", "Holdout"), ("crossroute", "Cross-route")):
        for i, ax in enumerate(AXES):
            s = d["axes"][ax][k]["summary"][theta]
            L.append(f"{name if i == 0 else ''} & {ax} & {s['n_runs_sustained_crossing']} & {s['n_detected_sustained']} & "
                     f"{f3(s['delay_sustained_median_s'], 2)} & {f3(s['delay_sustained_max_s'], 2)} & "
                     f"{f3(s['sustained_frac_within_1s'], 2)} & {f3(s['healthy_anchor_FA_rate'])} \\\\")
        if k == "holdout":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "alarm_delay.tex").write_text("\n".join(L) + "\n")
    # auxiliary: transient-crossing summary (for the text)
    tr = {}
    for k in ("holdout", "crossroute"):
        for ax in AXES:
            s = d["axes"][ax][k]["summary"][theta]
            tr[f"{k}/{ax}"] = {"n_cross": s["n_runs_crossing"], "missed": s["n_missed"], "median": s["delay_post_median_s"], "max": s["delay_post_max_s"]}
    (OUT / "alarm_delay_transient.json").write_text(json.dumps(tr, indent=1))


def capraw():
    d = json.loads((RES / "coverage/cap_vs_raw_coverage.json").read_text())
    rows = []
    for ax in AXES:
        for k in ("devval", "holdout", "crossroute"):
            s = d["axes"][ax]["sets"][k]
            rows.append(f"{ax}/{k}: n={s['n']} Y>t_max {s['n_Y_over_tmax']} ({s['frac_Y_over_tmax']*100:.2f}%), cov raw {s['cov_raw']} capped {s['cov_capped']}, bound>t_max {s['frac_bound_over_tmax']*100:.2f}%, max bound {s['max_bound']} vs t_max {d['axes'][ax]['t_max']}")
    (OUT / "capraw_summary.txt").write_text("\n".join(rows) + "\n")


def seeds():
    """Both final sets (holdout, cross-route) stacked as two blocks in one column (seeds_full.tex).
    The holdout-only form (four metrics) is still written to seeds.tex."""
    f = SV / "seed_variability_summary.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    names = [("cov", "Coverage"), ("cov_mid", "\\quad mid"), ("ECE_T1", "ECE at $T_1$"), ("twIBS", "twIBS")]
    names_full = names + [("AUROC_T1", "AUROC at $T_1$")]

    def rows(setn, nm):
        out = []
        for key, lab in nm:
            rf, rs = [lab, "frozen"], ["", "seeds"]
            for ax in AXES:
                t = d["table"][setn][ax].get(key); nd = 4 if key in ("twIBS", "AUROC_T1") else 3
                rf.append(cov3(t["frozen"], N_ANCH[setn]) if key == "cov" else f3(t["frozen"], nd)); rs.append(f"{t['mean']:.{nd}f}$\\pm${t['sd']:.{nd}f}")
            out.append(" & ".join(rf) + r" \\"); out.append(" & ".join(rs) + r" \\")
        return out

    L = [r"\begin{table}[!t]",
         r"\caption{Seed variability of the headline holdout metrics. The frozen model (seed 20260803) is the one reported elsewhere; five additional seeds re-run the entire registered pipeline (subsampling and cluster-fold seeds, $\lambda$ selection, isotonic calibration, and conformal shift). Entries for the additional seeds are mean $\pm$ SD.}",
         r"\label{tab:seeds}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llccc}", r"\toprule", r"Metric & & lon & lat & yaw \\", r"\midrule"]
    L += rows("holdout", names)
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "seeds.tex").write_text("\n".join(L) + "\n")

    L = [r"\begin{table}[H]",
         r"\caption{Seed variability on both final evaluation sets. The frozen model (seed 20260803) is the one reported elsewhere; five additional seeds re-run the entire registered pipeline (subsampling and cluster-fold seeds, $\lambda$ selection, isotonic calibration, and conformal shift). Entries for the additional seeds are mean $\pm$ SD over the five seeds. Raw monotonicity violations are zero for every seed, axis, and set.}",
         r"\label{tab:seeds}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llccc}", r"\toprule", r"Metric & & lon & lat & yaw \\", r"\midrule",
         r"\multicolumn{5}{l}{\textit{In-route holdout (routeA)}} \\"]
    L += rows("holdout", names_full)
    L += [r"\midrule", r"\multicolumn{5}{l}{\textit{Cross-route (routeB)}} \\"]
    L += rows("crossroute", names_full)
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "seeds_full.tex").write_text("\n".join(L) + "\n")


def latency():
    pass


def timing(theta="0.5"):
    fl, fd = RES / "timing/latency.json", RES / "timing/alarm_delay.json"
    if not (fl.exists() and fd.exists()):
        return
    L_ = json.loads(fl.read_text()); D = json.loads(fd.read_text())
    L = [r"\begin{table}[H]",
         r"\caption{Operational timing. (Top) Wall-clock latency per anchor for the three axis models evaluated sequentially on the desktop CPU of Sec.~\ref{ssec:platform}, over 300 anchors: the per-threshold path used for all reported results (106 single-row predictions per model), the grid-batched path that evaluates all grid points in one call per model with numerically identical outputs, and the same path restricted to one thread. (Bottom) Delay from the first sustained crossing of $T_{1,a}$ (error above $T_{1,a}$ for at least the $1$\,s label window) to the first anchor with calibrated $S_a(x,T_{1,a})\ge0.5$ on the final evaluation runs; negative values mean the alarm preceded the crossing, which the $1$\,s look-ahead label permits. Ep.: sustained episodes, all detected. FA: fraction of anchors with window error $Y_a \le T_{1,a}$ and an alarm.}",
         r"\label{tab:timing}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Three axes, sequential [ms] & Mean & p50 & p99 & Max & \\", r"\midrule"]
    names = {"core_outputs_as_reported": "Per-threshold (as reported)", "grid_batched": "Grid-batched", "grid_batched_1thread": "Grid-batched, one thread"}
    for key, lab in names.items():
        t = L_["three_axes_sequential_ms"][key]
        L.append(f"{lab} & {t['mean']:.1f} & {t['p50']:.1f} & {t['p99']:.1f} & {t['max']:.1f} & \\\\")
    L += [r"\midrule", r"Alarm delay [s] & Ep. & Min & Median & Max & FA \\", r"\midrule"]
    for k, name in (("holdout", "Holdout"), ("crossroute", "Cross-route")):
        for i_, ax in enumerate(AXES):
            s_ = D["axes"][ax][k]["summary"][theta]
            dmin = [r_[f"delay_sustained@{theta}"] for r_ in D["axes"][ax][k]["runs"] if r_.get(f"delay_sustained@{theta}") is not None]
            L.append(f"{name + ', ' if i_ == 0 else ''}{ax} & {s_['n_runs_sustained_crossing']} & {f3(min(dmin), 2) if dmin else '--'} & {f3(s_['delay_sustained_median_s'], 2)} & {f3(s_['delay_sustained_max_s'], 2)} & {f3(s_['healthy_anchor_FA_rate'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "timing.tex").write_text("\n".join(L) + "\n")


def rearranged_survival():
    fr, fs = RES / "baselines/rearranged_qmulti.json", RES / "baselines/survival_stacking.json"
    if not (fr.exists() and fs.exists()):
        return
    dr, ds = json.loads(fr.read_text()), json.loads(fs.read_text())
    ho = json.loads((HERE / "results/evaluation/holdout_eval.json").read_text())
    cr = json.loads((HERE / "results/evaluation/crossroute_eval.json").read_text())
    L = [r"\begin{table*}[!t]",
         r"\caption{Baselines from the survival and quantile literature on the same data and splits: survival stacking (discrete-hazard classification on the 41-point threshold grid; the curve is a cumulative product and is monotone by construction) and rearranged multi-quantile regression (nine quantile GBDTs, sorted per anchor; monotone after rearrangement). Both use the same GBDT hyperparameters and seed as \meld{} and the same conformal-shift fitting procedure for the bound. Raw quantile crossing is the fraction of adjacent quantile pairs that cross before rearrangement. Values are point estimates; no significance is claimed.}",
         r"\label{tab:litbase}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{Holdout} & \multicolumn{3}{c}{Cross-route} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Method & Metric & lon & lat & yaw & lon & lat & yaw \\", r"\midrule"]
    def meld(setn, ax, key):
        s = (ho if setn == "holdout" else cr)["axes"][ax]["all"]
        return {"twIBS": s["twIBS"]["MELD"], "auroc": s["auroc_T1"]["MELD"], "cov": s["q90_coverage"]["MELD_conf"],
                "mid": s["cov_by_band"].get("middle"), "ece": s["ECE_pT1"], "width": s["q90_coverage"]["MELD_mean_asinh_width"]}[key]
    for lab, key, nd in (("twIBS", "twIBS", 4), ("AUROC at $T_1$", "auroc", 3), ("ECE", "ece", 3), ("Coverage", "cov", 3), ("Width (asinh)", "width", 2)):
        L.append(" & ".join(["\\meld{}" if key == "twIBS" else "", lab] + [(cov3(meld(s, ax, key), N_ANCH[s]) if key == "cov" else f3(meld(s, ax, key), nd)) for s in ("holdout", "crossroute") for ax in AXES]) + r" \\")
    L.append(r"\midrule")
    for lab, key, nd in (("twIBS", "twIBS", 4), ("AUROC at $T_1$", "auroc_T1", 3), ("ECE", "ECE_pairs", 3), ("Coverage", "q90_coverage_conformal", 3), ("Width (asinh)", "mean_width_asinh", 2)):
        vals = []
        for s in ("holdout", "crossroute"):
            for ax in AXES:
                st = ds["axes"][ax]["sets"][s]
                vals.append(cov3(st[key], N_ANCH[s]) if key == "q90_coverage_conformal" else f3(st["cov_by_band"].get("middle") if key == "mid" else st[key], nd))
        L.append(" & ".join(["Survival stacking" if key == "twIBS" else "", lab] + vals) + r" \\")
    L.append(r"\midrule")
    for lab, key, nd in (("Raw q.\\ crossing", "cross", 3), ("twIBS", "twIBS", 4), ("AUROC at $T_1$", "auroc_T1", 3), ("ECE", "ECE_pairs", 3), ("Coverage", "q90_coverage_conformal", 3), ("Width (asinh)", "mean_width_asinh", 2)):
        vals = []
        for s in ("holdout", "crossroute"):
            for ax in AXES:
                st = dr["axes"][ax]["sets"][s]
                vals.append(cov3(st["variants"]["sorted"][key], N_ANCH[s]) if key == "q90_coverage_conformal" else f3(st["quantile_crossing_rate_adjacent"] if key == "cross" else st["variants"]["sorted"][key], nd))
        L.append(" & ".join(["Rearranged Q-multi" if key == "cross" else "", lab] + vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "litbase.tex").write_text("\n".join(L) + "\n")


def featprune():
    f = RES / "ablations/feature_groups.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    if "full_120" not in d["configs"] or not all(ax in d["configs"]["full_120"] for ax in AXES):
        return
    base = d["configs"]["full_120"]
    L = [r"\begin{table}[!t]",
         r"\caption{Grouped feature-pruning ablation (holdout). Each row retrains the registered pipeline without one statistic group (10 columns: 5 signals $\times$ 2 windows) or without the exactly redundant columns (dedup: most-recent value of the 8\,s window and the range, 15 columns). Entries are differences from the full 120-d model; negative twIBS and positive AUROC/coverage are better.}",
         r"\label{tab:featprune}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.4pt}",
         r"\begin{tabular}{lrccccccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{$\Delta$twIBS ($\times 10^{-3}$)} & \multicolumn{3}{c}{$\Delta$AUROC ($\times 10^{-3}$)} & \multicolumn{3}{c}{$\Delta$cov.\ mid} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"Config & $d$ & lon & lat & yaw & lon & lat & yaw & lon & lat & yaw \\", r"\midrule"]
    for name, cfg in d["configs"].items():
        if name == "full_120" or not all(ax in cfg for ax in AXES):
            continue
        row = [name.replace("_", "\\_"), str(cfg["n_features"])]
        row += [f"{(cfg[ax]['holdout']['twIBS_MELD'] - base[ax]['holdout']['twIBS_MELD'])*1e3:+.1f}" for ax in AXES]
        row += [f"{(cfg[ax]['holdout']['auroc_T1_MELD'] - base[ax]['holdout']['auroc_T1_MELD'])*1e3:+.1f}" for ax in AXES]
        row += [f"{(cfg[ax]['holdout']['cov_by_band'].get('middle', np.nan) - base[ax]['holdout']['cov_by_band'].get('middle', np.nan)):+.3f}" for ax in AXES]
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "featprune.tex").write_text("\n".join(L) + "\n")


def dyncov():
    f = RES / "stress/stress_eval.json"
    if not f.exists():
        return
    d = json.loads(f.read_text()); G = d["groups"].get("A_laplace")
    if not G:
        return
    L = [r"\begin{table}[!t]",
         r"\caption{Dynamic covariance of the matcher versus the \meld{} bound on the same runs. The 12 holdout recipes were re-executed with Autoware's Hessian-based (Laplace-approximation) covariance enabled; its scale factor ($30.4$) was fixed on an undegraded run so that the healthy median equals the configured floor, and was not changed afterwards. With the default scale the estimate never leaves the floor and equals BL-3$\sigma$. ``Scaled Hessian'' is the $3\sigma$ bound from the scaled estimate rotated into the heading frame; the yaw variance is not estimated by the matcher and stays at its configured value. AUROC and Spearman $\rho$ use $\sigma$ as the score; width is the mean bound in $\operatorname{asinh}$ units.}",
         r"\label{tab:dyncov}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
         r"\begin{tabular}{llccc}", r"\toprule", r"Axis & Quantity & Fixed $3\sigma$ & Scaled Hessian $3\sigma$ & \meld{} \\", r"\midrule"]
    for ax in AXES:
        g = G[ax]; l = g["laplace3s"]
        def c(dct, k):
            v = dct.get(k); return "--" if v is None else f"{v:.3f}"
        rows = [("coverage, all", f"{l['cov_fixed3s']:.3f}", f"{l['cov']:.3f}", f"{g['cov']:.3f}"),
                ("\\quad normal", c(l["cov_fixed3s_by_band"], "normal"), c(l["cov_by_band"], "normal"), c(g["cov_by_band"], "normal")),
                ("\\quad mid", c(l["cov_fixed3s_by_band"], "middle"), c(l["cov_by_band"], "middle"), c(g["cov_by_band"], "middle")),
                ("\\quad divergent", c(l["cov_fixed3s_by_band"], "div"), c(l["cov_by_band"], "div"), c(g["cov_by_band"], "div")),
                ("AUROC at $T_1$", "0.500", "--" if l["auroc_T1_sigma_as_score"] is None else f"{l['auroc_T1_sigma_as_score']:.3f}", f"{g['auroc_T1']:.3f}"),
                ("Spearman $\\rho$", "--", "--" if ax == "yaw" else f"{l['spearman_sigma_vs_err']:.2f}", "--"),
                ("width (asinh)", "--", f"{l['mean_width_asinh']:.2f}", f"{l['MELD_mean_width_asinh']:.2f}")]
        for i, r in enumerate(rows):
            L.append(" & ".join([ax if i == 0 else "", *r]) + r" \\")
        if ax != "yaw":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "dyncov.tex").write_text("\n".join(L) + "\n")


def stress():
    """Stress sets, two panels: (A) speed profiles, (B) high-intensity recipes; frozen models on rerun_features.npz."""
    import sys
    f = RES / "stress/stress_features.npz"
    if not f.exists():
        return
    sys.path.insert(0, str(HERE / "models/features")); sys.path.insert(0, str(HERE / "models/meldloc"))
    import meldloc_core as mc
    F = np.load(f); E = json.loads((RES / "stress/stress_eval.json").read_text())
    keys = sorted({k[:-3] for k in F.files if k.endswith("__X")})
    M = {ax: mc.MeldLoc.load(HERE / f"models/frozen/meldloc_{ax}.joblib") for ax in AXES}
    def pooled(ks):
        X = np.concatenate([F[k + "__X"] for k in ks]); Y = np.concatenate([F[k + "__Y"] for k in ks]); out = {}
        for ai, ax in enumerate(AXES):
            y = Y[:, ai]; t1, w3 = mc.T1[ax], mc.W3[ax]; o = M[ax].outputs(X); q, pt = o["q90_conf"], o["p_T1"]
            mid = (y > t1) & (y <= w3); lab = (y > t1).astype(int)
            out[ax] = {"n": int(len(y)), "cov": float((y <= q).mean()), "cov_mid": float((y[mid] <= q[mid]).mean()) if mid.sum() >= 20 else None,
                       "n_mid": int(mid.sum()), "FA": float((pt[y <= t1] >= 0.5).mean()) if (y <= t1).any() else None,
                       "auroc": mc.auroc(pt, lab) if 20 <= lab.sum() <= len(lab) - 20 else None, "viol": float(o["viol"]),
                       "p_T1_mid": float(pt[mid].mean()) if mid.any() else None}
        return out
    groups = {"std_h": [k for k in keys if k.startswith("healthy_std_fixed")], "fast_h": [k for k in keys if k.startswith("healthy_fast_fixed")],
              "aggr_h": [k for k in keys if k.startswith("healthy_aggr_fixed")],
              "fast_d": [k for k in keys if "_fast_fixed" in k and not k.startswith("healthy")],
              "aggr_d": [k for k in keys if "_aggr_fixed" in k and not k.startswith("healthy")],
              "point": [k for k in keys if k.startswith(("st_rd_95", "st_rc_8m", "st_sr_right170"))],
              "geom": [k for k in keys if k.startswith(("st_ls14", "st_ys26", "st_yr_3p5"))],
              "yr35": [k for k in keys if k.startswith("st_yr_3p5")]}
    P = {g: pooled(ks) for g, ks in groups.items() if ks}
    off = E["lon_offset_by_profile_healthy"]; off0 = E["lon_offset_frozen_routeA"]
    def tri(g, key, nd=3):
        return " & ".join(f3(P[g][ax][key], nd) for ax in AXES)
    L = [r"\begin{table*}[!t]",
         r"\caption{Evaluation-only stress sets on routeA scored with the frozen models (no retraining; not part of any cohort). (Top) Speed profiles: standard (maximum $4.9$\,m/s), fast (reference speed $\times1.35$, maximum $6.4$\,m/s), aggressive (reference alternating between $5.5$ and $2.0$\,m/s every $60$\,m); two undegraded runs per profile and four degraded recipes (lateral shear, range clip, yaw shear, dropout) for the fast and aggressive profiles. \capDelta{}offset: change of the healthy-run longitudinal offset relative to the frozen value. (Bottom) Intensities at or beyond the upper end of the training cohort: point loss (dropout $0.95$, range clip $8$\,m, sector removal $170^{\circ}$; the training runs reach $0.9$, $10$\,m, and $140^{\circ}$, except one run that reaches $180^{\circ}$ only in its final segment) and geometric (lateral shear $1.4$\,m, yaw shear $2.6^{\circ}$, yaw rotation $3.5^{\circ}$), with the rigid rotation also shown alone. Cov.: coverage of the $90\%$ bound; mid: coverage in the mid band; FA: fraction of anchors with window error $Y_a \le T_{1,a}$ and $S_a(x,T_{1,a})\ge0.5$; AUROC at $T_1$. Dashes: fewer than 20 anchors or a single class. Raw monotonicity violations are zero on every run.}",
         r"\label{tab:stress}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.2pt}",
         r"\begin{tabular}{lrcccccccccc}", r"\toprule",
         r"Speed profile & Runs & $\Delta$offset [m] & \multicolumn{3}{c}{Healthy FA (lon/lat/yaw)} & \multicolumn{3}{c}{Degraded cov.\ (lon/lat/yaw)} & \multicolumn{3}{c}{Degraded AUROC (lon/lat/yaw)} \\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}", r"\midrule"]
    for g, name, gd in (("std_h", "Standard", None), ("fast_h", "Fast", "fast_d"), ("aggr_h", "Aggressive", "aggr_d")):
        prof = {"std_h": "std", "fast_h": "fast", "aggr_h": "aggr"}[g]
        n = len(groups[g]) + (len(groups[gd]) if gd else 0)
        row = [name, str(n), f"{off[prof]['median'] - off0:+.3f}", tri(g, "FA")]
        row += [tri(gd, "cov"), tri(gd, "auroc")] if gd else ["-- & -- & --", "-- & -- & --"]
        L.append(" & ".join(row) + r" \\")
    L += [r"\midrule",
          r"High-intensity stress & Runs & & \multicolumn{3}{c}{Cov.\ (lon/lat/yaw)} & \multicolumn{3}{c}{mid cov.\ (lon/lat/yaw)} & \multicolumn{3}{c}{AUROC (lon/lat/yaw)} \\",
          r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}", r"\midrule"]
    for g, name in (("point", "Point loss"), ("geom", "Geometric"), ("yr35", "\\quad rigid yaw rotation $3.5^{\\circ}$")):
        L.append(" & ".join([name, str(len(groups[g])), "", tri(g, "cov"), tri(g, "cov_mid"), tri(g, "auroc")]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "stress.tex").write_text("\n".join(L) + "\n")
    # single-column transposed form (same values, axes as rows)
    cap = L[1]
    S = [r"\begin{table}[H]", cap, r"\label{tab:stress}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Speed profile & Standard & Fast & Aggressive \\", r"\midrule"]
    prof_cols = (("std_h", "std", None), ("fast_h", "fast", "fast_d"), ("aggr_h", "aggr", "aggr_d"))
    S.append("Runs & " + " & ".join(str(len(groups[g]) + (len(groups[gd]) if gd else 0)) for g, _, gd in prof_cols) + r" \\")
    S.append(r"$\Delta$offset [m] & " + " & ".join(f"{off[pr]['median'] - off0:+.3f}" for _, pr, _ in prof_cols) + r" \\")
    for ax in AXES:
        S.append(f"Healthy FA, {ax} & " + " & ".join(f3(P[g][ax]["FA"]) for g, _, _ in prof_cols) + r" \\")
    for key, lab in (("cov", "Degraded cov."), ("auroc", "Degraded AUROC")):
        for ax in AXES:
            S.append(f"{lab}, {ax} & " + " & ".join((f3(P[gd][ax][key]) if gd else "--") for _, _, gd in prof_cols) + r" \\")
    S += [r"\midrule", r"High-intensity stress & Point loss & Geometric & Rigid yaw $3.5^{\circ}$ \\", r"\midrule"]
    hi = ("point", "geom", "yr35")
    S.append("Runs & " + " & ".join(str(len(groups[g])) for g in hi) + r" \\")
    for key, lab in (("cov", "Cov."), ("cov_mid", "Mid cov."), ("auroc", "AUROC")):
        for ax in AXES:
            S.append(f"{lab}, {ax} & " + " & ".join(f3(P[g][ax][key]) for g in hi) + r" \\")
    S += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "stress_1col.tex").write_text("\n".join(S) + "\n")
    (RES / "stress/stress_summary.json").write_text(json.dumps(P, indent=1))
    (RES / "stress/stress_groups.json").write_text(json.dumps(groups, indent=1))


def markdown_tables():
    """Full tables (markdown): all seeds, feature pruning, per-axis delays, per-axis stress results."""
    md = ["# Full result tables for the response letter (auto-generated)\n"]
    f = SV / "seed_variability_summary.json"
    if f.exists():
        d = json.loads(f.read_text()); md.append("## Seed variability (frozen seed + five additional seeds)\n")
        for setn in ("holdout", "crossroute"):
            md.append(f"### {setn}\n\n| axis | metric | frozen | mean ± SD | min – max |\n|---|---|---|---|---|")
            for ax in AXES:
                for k, t in d["table"][setn][ax].items():
                    if k == "baselines" or not isinstance(t, dict) or "mean" not in t:
                        continue
                    md.append(f"| {ax} | {k} | {t['frozen']} | {t['mean']:.4f} ± {t['sd']:.4f} | {t['min']:.4f} – {t['max']:.4f} |")
            md.append("")
    f = RES / "ablations/feature_groups.json"
    if f.exists():
        d = json.loads(f.read_text()); base = d["configs"]["full_120"]
        md.append("## Grouped feature-pruning ablation (holdout, same seed; differences from the full 120-d model)\n")
        md.append("| config | d | ΔtwIBS lon/lat/yaw (×1e-3) | ΔAUROC lon/lat/yaw (×1e-3) | Δmid-coverage lon/lat/yaw |\n|---|---|---|---|---|")
        for name, cfg in d["configs"].items():
            if name == "full_120":
                continue
            tw = "/".join(f"{(cfg[ax]['holdout']['twIBS_MELD'] - base[ax]['holdout']['twIBS_MELD'])*1e3:+.1f}" for ax in AXES)
            au = "/".join(f"{(cfg[ax]['holdout']['auroc_T1_MELD'] - base[ax]['holdout']['auroc_T1_MELD'])*1e3:+.1f}" for ax in AXES)
            cm = "/".join(f"{(cfg[ax]['holdout']['cov_by_band'].get('middle', float('nan')) - base[ax]['holdout']['cov_by_band'].get('middle', float('nan'))):+.3f}" for ax in AXES)
            md.append(f"| {name} | {cfg['n_features']} | {tw} | {au} | {cm} |")
        md.append("")
    f = RES / "timing/latency.json"
    if f.exists():
        d = json.loads(f.read_text()); md.append("## Single-anchor latency per axis (ms, 300 anchors, desktop CPU)\n")
        md.append("| path | axis | mean | p50 | p99 | max |\n|---|---|---|---|---|---|")
        for key in ("core_outputs_as_reported", "grid_batched", "grid_batched_1thread"):
            for ax in AXES:
                t = d["axes"][ax][key]; md.append(f"| {key} | {ax} | {t['mean_ms']} | {t['p50_ms']} | {t['p99_ms']} | {t['max_ms']} |")
            t = d["three_axes_sequential_ms"][key]; md.append(f"| {key} | 3 axes seq. | {t['mean']} | {t['p50']} | {t['p99']} | {t['max']} |")
        md.append("")
    f = RES / "timing/alarm_delay.json"
    if f.exists():
        d = json.loads(f.read_text()); md.append("## Alarm delay, all thresholds (sustained and transient crossings)\n")
        md.append("| set | axis | θ | sustained ep. | detected | median [s] | q25–q75 | max | within 1 s | transient first-crossings | missed | median [s] | FA rate |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k in ("holdout", "crossroute"):
            for ax in AXES:
                for th, s_ in d["axes"][ax][k]["summary"].items():
                    md.append(f"| {k} | {ax} | {th} | {s_['n_runs_sustained_crossing']} | {s_['n_detected_sustained']} | {s_['delay_sustained_median_s']} | {s_['delay_sustained_q25_q75_s']} | {s_['delay_sustained_max_s']} | {s_['sustained_frac_within_1s']} | {s_['n_runs_crossing']} | {s_['n_missed']} | {s_['delay_post_median_s']} | {s_['healthy_anchor_FA_rate']} |")
        md.append("")
    f = RES / "stress/stress_summary.json"
    if f.exists():
        d = json.loads(f.read_text()); md.append("## Stress sets, per axis (frozen models)\n")
        md.append("| group | axis | n | coverage | mid coverage | n mid | FA@0.5 | AUROC | mean P(Y>T1) in mid | viol |\n|---|---|---|---|---|---|---|---|---|---|")
        for g, v in d.items():
            for ax in AXES:
                a = v[ax]; md.append(f"| {g} | {ax} | {a['n']} | {a['cov']:.3f} | {a['cov_mid'] if a['cov_mid'] is None else round(a['cov_mid'],3)} | {a['n_mid']} | {a['FA'] if a['FA'] is None else round(a['FA'],3)} | {a['auroc'] if a['auroc'] is None else round(a['auroc'],3)} | {a['p_T1_mid'] if a['p_T1_mid'] is None else round(a['p_T1_mid'],2)} | {a['viol']:.6f} |")
        md.append("")
    (HERE / "reports/tables.md").write_text("\n".join(md))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (condcov, alarm_delay, capraw, seeds, timing, rearranged_survival, featprune, dyncov, stress, markdown_tables):
        try:
            fn(); print("ok", fn.__name__)
        except Exception as e:
            print("skip", fn.__name__, repr(e))
