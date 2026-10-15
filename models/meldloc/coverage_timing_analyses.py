#!/usr/bin/env python3
"""Analyses on the frozen model outputs only (no re-training, no new simulation runs).

  condcov : conditional coverage by bins of the predicted alarm probability P(Y>T1)
            + per-bin (Mondrian) conformal shift experiment           
  capraw  : capped vs raw coverage; anchors above t_max; bounds above t_max
  delay   : delay from the first T1 crossing to the first alarm; healthy false-alarm rates
  latency : single-anchor latency mean/p50/p99/max, three axes sequentially, grid-batched path

The frozen models are read only. Output: the functional sub-folders of results/ (see models/README.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "models/features"))
sys.path.insert(0, str(HERE / "models/meldloc"))
import build_features as hc  # noqa: E402
import meldloc_core as mc  # noqa: E402

MODEL_DIR = HERE / "models/frozen"
CACHE = HERE / "results/features/development_features.npz"
SETS = {"holdout": (HERE / "results/features/holdout_features.npz",
                    HERE / "data/logs/holdout/ledger.csv", "routeA"),
        "crossroute": (HERE / "results/features/crossroute_features.npz",
                       HERE / "data/logs/crossroute/ledger.csv", "routeB_town02_S1")}
RES = HERE / "results"
AXES = ("lon", "lat", "yaw")
P_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]


def load_models():
    return {ax: mc.MeldLoc.load(MODEL_DIR / f"meldloc_{ax}.joblib") for ax in AXES}


def load_sets():
    Z = np.load(CACHE, allow_pickle=True)
    va = Z["split"].astype(str) == "validation"
    grp = Z["recipe_group"].astype(str) if "recipe_group" in Z else Z["cluster"].astype(str)
    d = {"devval": dict(X=Z["X"].astype(np.float64)[va], Y=Z["Y"].astype(np.float64)[va],
                        family=Z["family"].astype(str)[va], group=grp[va],
                        scenario=Z["scenario"].astype(str)[va], t_anchor=Z["t_anchor"].astype(np.float64)[va])}
    for k, (npz, _, _) in SETS.items():
        Zh = np.load(npz, allow_pickle=True)
        d[k] = dict(X=Zh["X"].astype(np.float64), Y=Zh["Y"].astype(np.float64),
                    family=Zh["family"].astype(str), group=Zh["group"].astype(str),
                    scenario=Zh["scenario"].astype(str), t_anchor=Zh["t_anchor"].astype(np.float64))
    return d


def frozen_outputs(models, sets, log):
    """Cached outputs per set and axis (p_T1, q90_raw, q90_conf, ...)."""
    for _d in ('coverage', 'timing'): (RES / _d).mkdir(parents=True, exist_ok=True)
    res = {}
    for k, D in sets.items():
        f = RES / f"coverage/frozen_outputs_{k}.npz"
        if f.exists():
            Zo = np.load(f, allow_pickle=True)
            res[k] = {ax: {q: Zo[f"{ax}_{q}"] for q in ("p_T1", "p_T2", "p_3W", "q90_raw", "q90_conf", "e_med", "R")}
                      for ax in AXES}
            continue
        res[k] = {}
        save = {}
        for ax in AXES:
            o = models[ax].outputs(D["X"])
            res[k][ax] = {q: o[q] for q in ("p_T1", "p_T2", "p_3W", "q90_raw", "q90_conf", "e_med", "R")}
            for q, v in res[k][ax].items():
                save[f"{ax}_{q}"] = v
            log(f"{k}/{ax}: outputs computed (viol {o['viol']:.4%})")
        np.savez_compressed(f, **save)
    return res


def wquant(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)])


def binom_ci(k, n):
    if n == 0:
        return [None, None]
    p = k / n; z = 1.96
    den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def cmd_condcov(models, sets, outs, log):
    """Conditional coverage by predicted-alarm-probability bin, group-conditional test, Mondrian shift."""
    res = {"bins_on": "calibrated P(Y>T1) (online-available, no ground truth)",
           "bin_edges": P_BINS[:-1] + [1.0], "axes": {}}
    for ax in AXES:
        M = models[ax]; ai = AXES.index(ax); t1, w3 = mc.T1[ax], mc.W3[ax]
        # Mondrian shift: per-bin delta_b on dev-val (group-weighted 90th percentile); bins with n<300 use the global delta
        Dv = sets["devval"]; yv = Dv["Y"][:, ai]; pv = outs["devval"][ax]["p_T1"]
        rv = np.arcsinh(yv) - np.arcsinh(outs["devval"][ax]["q90_raw"])
        gw = mc.group_equal_weights(Dv["group"])
        bv = np.digitize(pv, P_BINS) - 1
        delta_b = {}
        for b in range(len(P_BINS) - 1):
            m = bv == b
            delta_b[b] = wquant(rv[m], gw[m], 0.9) if m.sum() >= 300 else M.conf_shift
        A = {"frozen_delta": round(M.conf_shift, 4),
             "mondrian_delta_by_bin": {f"[{P_BINS[b]:.1f},{min(P_BINS[b+1],1):.1f})": round(v, 4) for b, v in delta_b.items()},
             "sets": {}}
        for k in ("devval", "holdout", "crossroute"):
            D = sets[k]; y = D["Y"][:, ai]; O = outs[k][ax]
            p = O["p_T1"]; q_frozen = O["q90_conf"]
            b_idx = np.digitize(p, P_BINS) - 1
            q_mond = np.sinh(np.arcsinh(O["q90_raw"]) + np.array([delta_b[b] for b in b_idx]))
            band = np.where(y <= t1, "normal", np.where(y <= w3, "middle", "div"))
            rows = []
            # group-conditional test: homogeneity of coverage across bins (chi-square, df=B-1)
            ks, ns = [], []
            for b in range(len(P_BINS) - 1):
                m = b_idx == b
                if m.sum() == 0:
                    continue
                cov_f = int((y[m] <= q_frozen[m]).sum()); cov_m = int((y[m] <= q_mond[m]).sum())
                ks.append(cov_f); ns.append(int(m.sum()))
                rows.append({"bin": f"[{P_BINS[b]:.1f},{min(P_BINS[b+1],1):.1f})", "n": int(m.sum()),
                             "mean_pT1": round(float(p[m].mean()), 3),
                             "empirical_exceed_T1": round(float((y[m] > t1).mean()), 3),
                             "cov_frozen": round(cov_f / m.sum(), 3), "cov_frozen_ci": binom_ci(cov_f, m.sum()),
                             "cov_mondrian": round(cov_m / m.sum(), 3),
                             "band_mix": {bb: int((band[m] == bb).sum()) for bb in ("normal", "middle", "div")}})
            ks, ns = np.array(ks), np.array(ns)
            p_pool = ks.sum() / ns.sum()
            chi2 = float((((ks - ns * p_pool) ** 2) / (ns * p_pool * (1 - p_pool) + 1e-12)).sum())
            df = max(len(ks) - 1, 1)
            try:
                from scipy.stats import chi2 as chi2d
                pval = float(chi2d.sf(chi2, df))
            except Exception:
                pval = None
            A["sets"][k] = {"n": int(len(y)), "marginal_cov_frozen": round(float((y <= q_frozen).mean()), 4),
                            "marginal_cov_mondrian": round(float((y <= q_mond).mean()), 4),
                            "mid_cov_frozen": round(float((y[band == "middle"] <= q_frozen[band == "middle"]).mean()), 3) if (band == "middle").sum() else None,
                            "mid_cov_mondrian": round(float((y[band == "middle"] <= q_mond[band == "middle"]).mean()), 3) if (band == "middle").sum() else None,
                            "min_bin_cov_frozen": round(float(min(r["cov_frozen"] for r in rows if r["n"] >= 30)), 3),
                            "min_bin_cov_mondrian": round(float(min(r["cov_mondrian"] for r in rows if r["n"] >= 30)), 3),
                            "homogeneity_chi2": round(chi2, 2), "df": df, "p_value": pval,
                            "bins": rows}
            log(f"{ax}/{k}: marg {A['sets'][k]['marginal_cov_frozen']} → mond {A['sets'][k]['marginal_cov_mondrian']} | "
                f"min-bin {A['sets'][k]['min_bin_cov_frozen']} → {A['sets'][k]['min_bin_cov_mondrian']} | "
                f"mid {A['sets'][k]['mid_cov_frozen']} → {A['sets'][k]['mid_cov_mondrian']} | chi2 p={pval}")
        res["axes"][ax] = A
    (RES / "coverage/conditional_coverage.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("-> conditional_coverage.json")


def cmd_capraw(models, sets, outs, log):
    res = {"definition": "coverage is computed against the raw (uncapped) error: 1{Y_raw <= q90_conf}; only training labels are capped",
           "axes": {}}
    for ax in AXES:
        M = models[ax]; ai = AXES.index(ax); A = {"t_max": round(M.t_max, 3), "sets": {}}
        for k in ("devval", "holdout", "crossroute"):
            y = sets[k]["Y"][:, ai]; q = outs[k][ax]["q90_conf"]; qr = outs[k][ax]["q90_raw"]
            over = y > M.t_max
            A["sets"][k] = {"n": int(len(y)), "n_Y_over_tmax": int(over.sum()),
                            "frac_Y_over_tmax": round(float(over.mean()), 5),
                            "cov_raw": round(float((y <= q).mean()), 4),
                            "cov_capped": round(float((np.minimum(y, M.t_max) <= q).mean()), 4),
                            "cov_among_over_tmax": round(float((y[over] <= q[over]).mean()), 3) if over.any() else None,
                            "frac_bound_over_tmax": round(float((q > M.t_max).mean()), 5),
                            "frac_raw_q90_at_tmax_grid_end": round(float(np.isclose(qr, M.t_grid[-1]).mean()), 5),
                            "max_bound": round(float(q.max()), 2), "max_Y": round(float(y.max()), 2)}
            log(f"{ax}/{k}: Y>t_max {over.sum()} ({over.mean():.3%}) cov raw {A['sets'][k]['cov_raw']} "
                f"capped {A['sets'][k]['cov_capped']} | bound>t_max {A['sets'][k]['frac_bound_over_tmax']:.3%}")
        res["axes"][ax] = A
    (RES / "coverage/cap_vs_raw_coverage.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("-> cap_vs_raw_coverage.json")


def run_dirs(ledger):
    rows = list(csv.DictReader(open(ledger)))
    return {f"{r['scenario_id']}_r{r['replicate']}": HERE / r["run_dir"] for r in rows
            if r["probe"] == "OK" and r["go_ok"] == "1"}


def cmd_delay(models, sets, outs, log, thetas=(0.3, 0.5, 0.7)):
    """Delay from the first T1 crossing (raw error series) to the first alarm (P(Y>T1) >= theta, 0.5 s anchors)."""
    res = {"alarm_rule": "calibrated P(Y>T1) >= theta at 0.5 s anchors; label window = [tau, tau+1 s]",
           "thetas": list(thetas), "axes": {}}
    for ax in AXES:
        ai = AXES.index(ax); t1 = mc.T1[ax]; A = {}
        for k, (_, ledger, route) in SETS.items():
            D = sets[k]; O = outs[k][ax]; dirs = run_dirs(ledger)
            off = hc.LON_OFFSETS[route] if ax == "lon" else 0.0
            per_run = []
            for s, d in sorted(dirs.items()):
                dd, why = hc.load_run(d)
                if dd is None:
                    per_run.append({"run": s, "status": why}); continue
                t = dd["timestamp"]; e = np.abs(dd[hc.COL[ax]] - off)
                cross = np.where(e > t1)[0]
                # sustained crossing: first time e>T1 holds for >= 1.0 s (the label window)
                above = e > t1; sus = None
                i0 = 0
                while i0 < len(above):
                    if above[i0]:
                        j = i0
                        while j + 1 < len(above) and above[j + 1]:
                            j += 1
                        if t[j] - t[i0] >= 1.0 - 1e-6:
                            sus = i0; break
                        i0 = j + 1
                    else:
                        i0 += 1
                m = D["scenario"] == s
                ta = D["t_anchor"][m]; p = O["p_T1"][m]; o = np.argsort(ta); ta, p = ta[o], p[o]
                row = {"run": s, "family": str(D["family"][m][0]) if m.any() else None,
                       "n_anchors": int(m.sum()), "first_anchor_s": round(float(ta[0] - t[0]), 1) if m.any() else None}
                if len(cross) == 0:
                    row["t_cross"] = None
                    for th in thetas:
                        row[f"false_alarm_anchors@{th}"] = int((p >= th).sum())
                    per_run.append(row); continue
                tc = float(t[cross[0]]); row["t_cross_s"] = round(tc - t[0], 1)
                row["cross_before_first_anchor"] = bool(m.any() and tc < ta[0])
                if sus is not None:
                    tcs = float(t[sus]); row["t_cross_sustained_s"] = round(tcs - t[0], 1)
                    for th in thetas:
                        al3 = np.where((p >= th) & (ta >= tcs - 1.0))[0]
                        row[f"delay_sustained@{th}"] = round(float(ta[al3[0]] - tcs), 2) if len(al3) else None
                else:
                    row["t_cross_sustained_s"] = None
                for th in thetas:
                    al = np.where(p >= th)[0]
                    if len(al) == 0:
                        row[f"delay@{th}"] = None; row[f"missed@{th}"] = True; continue
                    row[f"delay@{th}"] = round(float(ta[al[0]] - tc), 2)      # negative = alarm before the crossing
                    # first alarm after the crossing (tc - 1 s, allowing for the label window)
                    al2 = np.where((p >= th) & (ta >= tc - 1.0))[0]
                    row[f"delay_post@{th}"] = round(float(ta[al2[0]] - tc), 2) if len(al2) else None
                    row[f"missed@{th}"] = False
                per_run.append(row)
            summ = {}
            for th in thetas:
                dl = np.array([r[f"delay_post@{th}"] for r in per_run if r.get(f"delay_post@{th}") is not None])
                d0 = np.array([r[f"delay@{th}"] for r in per_run if r.get(f"delay@{th}") is not None])
                ncross = sum(1 for r in per_run if r.get("t_cross_s") is not None)
                ds = np.array([r[f"delay_sustained@{th}"] for r in per_run if r.get(f"delay_sustained@{th}") is not None])
                nsus = sum(1 for r in per_run if r.get("t_cross_sustained_s") is not None)
                summ[str(th)] = {"n_runs_crossing": ncross, "n_runs_sustained_crossing": nsus,
                                 "n_detected_sustained": int(len(ds)),
                                 "delay_sustained_median_s": round(float(np.median(ds)), 2) if len(ds) else None,
                                 "delay_sustained_q25_q75_s": [round(float(np.percentile(ds, 25)), 2), round(float(np.percentile(ds, 75)), 2)] if len(ds) else None,
                                 "delay_sustained_max_s": round(float(ds.max()), 2) if len(ds) else None,
                                 "sustained_frac_within_1s": round(float((ds <= 1.0).mean()), 3) if len(ds) else None,
                                 "sustained_frac_within_2s": round(float((ds <= 2.0).mean()), 3) if len(ds) else None,
                                 "n_detected": int(len(dl)), "n_missed": int(sum(1 for r in per_run if r.get(f"missed@{th}"))),
                                 "delay_post_median_s": round(float(np.median(dl)), 2) if len(dl) else None,
                                 "delay_post_q25_q75_s": [round(float(np.percentile(dl, 25)), 2), round(float(np.percentile(dl, 75)), 2)] if len(dl) else None,
                                 "delay_post_max_s": round(float(dl.max()), 2) if len(dl) else None,
                                 "frac_within_1s": round(float((dl <= 1.0).mean()), 3) if len(dl) else None,
                                 "frac_within_2s": round(float((dl <= 2.0).mean()), 3) if len(dl) else None,
                                 "first_alarm_before_cross_frac": round(float((d0 < 0).mean()), 3) if len(d0) else None,
                                 "healthy_anchor_FA_rate": round(float((O["p_T1"][D["Y"][:, ai] <= t1] >= th).mean()), 4),
                                 "never_cross_runs_FA_anchors": int(sum(r.get(f"false_alarm_anchors@{th}", 0) for r in per_run if r.get("t_cross_s") is None))}
            A[k] = {"summary": summ, "runs": per_run}
            log(f"{ax}/{k}: " + " | ".join(f"θ{th}: any-cross med {summ[str(th)]['delay_post_median_s']}s max {summ[str(th)]['delay_post_max_s']}s "
                                           f"miss {summ[str(th)]['n_missed']}/{summ[str(th)]['n_runs_crossing']} ; sustained({summ[str(th)]['n_runs_sustained_crossing']}) "
                                           f"med {summ[str(th)]['delay_sustained_median_s']}s max {summ[str(th)]['delay_sustained_max_s']}s "
                                           f"det {summ[str(th)]['n_detected_sustained']} ; FA {summ[str(th)]['healthy_anchor_FA_rate']}" for th in thetas))
        res["axes"][ax] = A
    (RES / "timing/alarm_delay.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("-> alarm_delay.json")


def grid_batched_outputs(M, x_row, num_threads=None):
    """Single anchor: evaluate the curve grid (41) and the R grid (65) in one matrix so F and G are called once each.
    Must be numerically identical to core.outputs (checked in cmd_latency)."""
    ax = M.axis; t1, t2, w3 = mc.T1[ax], mc.T2[ax], mc.W3[ax]
    at = float(np.arcsinh(M.t_max)); tg = M.t_grid
    r_grid = np.linspace(t1, w3, mc.N_RGRID)
    ts = np.concatenate([tg, r_grid])
    tf = (np.arcsinh(ts) / at).astype(np.float32)[:, None]
    Xa = np.repeat(x_row[M.xa_cols].astype(np.float32)[None, :], len(ts), axis=0)
    kw = {} if num_threads is None else {"num_threads": num_threads}
    z = M.F.predict(np.hstack([Xa, tf]), raw_score=True, **kw)
    if M.lam != 0.0:
        Xs = np.repeat(x_row[M.xs_cols].astype(np.float32)[None, :], len(ts), axis=0)
        z = z + M.lam * M.G.predict(np.hstack([Xs, tf]), raw_score=True, **kw)
    p = M.iso.predict(mc.sigmoid(z)).astype(np.float32)
    S = np.minimum.accumulate(p[:len(tg)])[None, :]
    S_R = np.minimum.accumulate(p[len(tg):])[None, :]
    e_med = mc.first_below(S, tg, 0.5); q90_raw = mc.first_below(S, tg, 0.10)
    return {"S": S, "e_med": e_med, "q90_raw": q90_raw,
            "q90_conf": np.sinh(np.arcsinh(q90_raw) + M.conf_shift),
            "p_T1": mc.prob_at(S, tg, t1), "p_T2": mc.prob_at(S, tg, t2), "p_3W": mc.prob_at(S, tg, w3),
            "R": np.trapezoid(S_R, r_grid, axis=1)}


def cmd_latency(models, sets, outs, log, n=300):
    X = sets["holdout"]["X"]; idx = np.linspace(0, len(X) - 1, n).astype(int)
    res = {"n_anchors_timed": n, "note": "single-anchor wall-clock per call; idle CPU required",
           "axes": {}, "three_axes_sequential_ms": {}}
    # numerical identity check
    for ax in AXES:
        M = models[ax]; md = 0.0
        for i in idx[:100]:
            a = M.outputs(X[i:i + 1]); b = grid_batched_outputs(M, X[i])
            md = max(md, *[float(np.abs(a[q] - b[q]).max()) for q in ("e_med", "q90_conf", "p_T1", "p_T2", "p_3W", "R")])
        res["axes"][ax] = {"grid_batched_max_abs_diff_vs_core": md}
        log(f"{ax}: grid-batched vs core max abs difference {md:.2e}")
    variants = {"core_outputs_as_reported": lambda M, x: M.outputs(x[None, :]),
                "grid_batched": lambda M, x: grid_batched_outputs(M, x),
                "grid_batched_1thread": lambda M, x: grid_batched_outputs(M, x, num_threads=1)}
    for name, fn in variants.items():
        per_ax = {}
        seq = []
        for ax in AXES:
            M = models[ax]; ts = []
            for _ in range(3):
                fn(M, X[idx[0]])                      # warm-up
            for i in idx:
                t0 = time.perf_counter(); fn(M, X[i]); ts.append((time.perf_counter() - t0) * 1e3)
            ts = np.array(ts)
            per_ax[ax] = {"mean_ms": round(float(ts.mean()), 2), "p50_ms": round(float(np.median(ts)), 2),
                          "p99_ms": round(float(np.percentile(ts, 99)), 2), "max_ms": round(float(ts.max()), 2)}
        for i in idx:
            t0 = time.perf_counter()
            for ax in AXES:
                fn(models[ax], X[i])
            seq.append((time.perf_counter() - t0) * 1e3)
        seq = np.array(seq)
        res["three_axes_sequential_ms"][name] = {"mean": round(float(seq.mean()), 2), "p50": round(float(np.median(seq)), 2),
                                                 "p99": round(float(np.percentile(seq, 99)), 2), "max": round(float(seq.max()), 2)}
        for ax in AXES:
            res["axes"][ax][name] = per_ax[ax]
        log(f"{name}: " + " ".join(f"{ax} {per_ax[ax]['mean_ms']}/{per_ax[ax]['p99_ms']}/{per_ax[ax]['max_ms']}" for ax in AXES) +
            f" | three axes sequential mean {res['three_axes_sequential_ms'][name]['mean']} p99 {res['three_axes_sequential_ms'][name]['p99']} ms")
    # batch throughput (reference)
    for ax in AXES:
        t0 = time.perf_counter(); models[ax].outputs(X[:2000]); dt = (time.perf_counter() - t0) * 1e3
        res["axes"][ax]["batched_ms_per_anchor_2000"] = round(dt / 2000, 3)
    try:
        import subprocess
        res["cpu"] = subprocess.run(["bash", "-c", "lscpu | grep 'Model name' | sed 's/.*: *//'"], capture_output=True, text=True).stdout.strip()
        res["load_1min"] = float(open("/proc/loadavg").read().split()[0])
    except Exception:
        pass
    (RES / "timing/latency.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("-> latency.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmds", nargs="+", choices=["condcov", "capraw", "delay", "latency"])
    a = ap.parse_args()
    t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    models = load_models(); sets = load_sets(); outs = frozen_outputs(models, sets, log)
    for c in a.cmds:
        {"condcov": cmd_condcov, "capraw": cmd_capraw, "delay": cmd_delay, "latency": cmd_latency}[c](models, sets, outs, log)


if __name__ == "__main__":
    main()
