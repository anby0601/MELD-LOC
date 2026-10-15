#!/usr/bin/env python3
"""Scoring of the evaluation-only re-runs (stress sets and Hessian covariance) with the frozen models.

Group A (laplace, std): the matcher's Hessian (Laplace) covariance as a 3-sigma bound vs the MELD bound on the same runs:
   coverage by regime, Spearman of sigma vs realized error, AUROC@T1 with sigma as score, fixed 3-sigma alongside.
   Autoware keeps the yaw covariance fixed, so yaw coincides with the fixed option.
Group B (profiles, fixed): per-profile healthy false alarms, longitudinal-offset re-estimation, coverage/AUROC/ECE of the degraded recipes.
Group C (high intensity, fixed, std): curve saturation (p_T1, p_3W, q90) and coverage.

Input: the stress and ndt_covariance ledgers (probe OK and go_ok=1 only). Output: results/stress/stress_eval.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "models/features"))
sys.path.insert(0, str(HERE / "models/meldloc"))
import build_features as hc  # noqa: E402
import meldloc_core as mc  # noqa: E402

MODEL_DIR = HERE / "models/frozen"
LEDGERS = [HERE / "data/logs/stress/ledger.csv", HERE / "data/logs/ndt_covariance/ledger.csv"]
OUT = HERE / "results/stress/stress_eval.json"
NPZ = HERE / "results/stress/stress_features.npz"
AXES = ("lon", "lat", "yaw")
FAMILY = {"ho_ls08": "lateral_shear", "ho_rc_early12": "range_clip", "ho_rd_6080": "random_dropout",
          "ho_rd_7090": "random_dropout", "ho_rd_gradsweep": "random_dropout", "ho_rd_temporal80": "random_dropout",
          "ho_rd_uniform60": "random_dropout", "ho_sr_rear120": "sector_remove", "ho_sr_right125": "sector_remove",
          "ho_sr_right140": "sector_remove", "ho_yr_1p2": "yaw_rotate", "ho_ys15": "yaw_shear", "healthy": "healthy",
          "st_rd_95": "random_dropout", "st_rc_8m": "range_clip", "st_sr_right170": "sector_remove",
          "st_ls14": "lateral_shear", "st_ys26": "yaw_shear", "st_yr_3p5": "yaw_rotate"}
FIXED_3S = {"lon": 0.45, "lat": 0.45, "yaw": 4.30}          # fixed 3-sigma constants (m / deg)


def read_cov(d: Path):
    """Covariance columns of the logger (sorted by timestamp), rotated into the heading frame: sigma_lon, sigma_lat, sigma_yaw."""
    rows = list(csv.DictReader(open(d / "long_route_pilot.csv")))
    t = np.array([float(r["timestamp"]) for r in rows])
    f = lambda k: np.array([float(r.get(k, "nan") or "nan") for r in rows])
    cxx, cyy, cxy, cyaw = f("ndt_cov_xx"), f("ndt_cov_yy"), f("ndt_cov_xy"), f("ndt_cov_yawyaw")
    gy = np.radians(f("gt_yaw"))
    c, s = np.cos(gy), np.sin(gy)
    var_lon = c * c * cxx + 2 * c * s * cxy + s * s * cyy
    var_lat = s * s * cxx - 2 * c * s * cxy + c * c * cyy
    o = np.argsort(t, kind="mergesort")
    return {"t": t[o], "sig_lon": np.sqrt(np.maximum(var_lon[o], 0)), "sig_lat": np.sqrt(np.maximum(var_lat[o], 0)),
            "sig_yaw": np.degrees(np.sqrt(np.maximum(cyaw[o], 0))), "exe_ms": f("ndt_exe_time_ms")[o]}


def extract(run_dir: Path, lon_off: float):
    dd, why = hc.load_run(run_dir)
    if dd is None:
        return None, why
    t = dd["timestamp"]
    hi = np.searchsorted(t, t + hc.H, side="left")
    lab = {}
    for ax in hc.AX:
        e = np.abs(dd[hc.COL[ax]] - (lon_off if ax == "lon" else 0.0))
        lab[ax] = np.array([e[i:max(j, i + 1)].max() for i, j in enumerate(hi)])
    X, Y, TA, IDX = [], [], [], []
    for tau in np.arange(t[0] + hc.HIST_MIN, t[-1] - hc.H, hc.STRIDE):
        i = np.searchsorted(t, tau, side="right") - 1
        if i < 0 or tau - t[i] > 0.5:
            continue
        f, bad = [], False
        for L in hc.SCALES:
            for s in hc.SIG:
                w = hc.window_stats(t, dd[s], tau - L, tau)
                if w is None:
                    bad = True; break
                f += w
            if bad:
                break
        if bad:
            continue
        X.append(f); Y.append([lab[ax][i] for ax in hc.AX]); TA.append(tau); IDX.append(i)
    return {"X": np.array(X, np.float64), "Y": np.array(Y, np.float64), "t_anchor": np.array(TA), "idx": np.array(IDX),
            "raw": dd, "t": t}, None


def main():
    models = {ax: mc.MeldLoc.load(MODEL_DIR / f"meldloc_{ax}.joblib") for ax in AXES}
    rows = [r for L in LEDGERS for r in csv.DictReader(open(L)) if r["go_ok"] == "1" and int(r["csv_rows"]) > 1000 and (r["probe"] == "OK" or r["covtype"] == "laplace_raw")]
    runs = {}
    for r in rows:
        key = f"{r['scenario_id']}_{r['profile']}_{r['covtype']}_r{r['replicate']}"
        runs[key] = r
    # 1) per-profile longitudinal offset from the healthy runs (compared with the frozen routeA offset)
    off_frozen = hc.LON_OFFSETS["routeA"]
    off_prof = {}
    for prof in ("std", "fast", "aggr"):
        res_all = []
        for key, r in runs.items():
            if r["scenario_id"] == "healthy" and r["profile"] == prof and r["covtype"] in ("fixed", "laplace"):
                dd, _ = hc.load_run(HERE / r["run_dir"])
                if dd is not None:
                    res_all.append(dd["raw_longitudinal_residual"])
        if res_all:
            v = np.concatenate(res_all)
            off_prof[prof] = {"n_rows": int(len(v)), "median": float(np.median(v)),
                              "q10": float(np.quantile(v, .1)), "q90": float(np.quantile(v, .9)),
                              "delta_vs_frozen_m": float(np.median(v) - off_frozen)}
    out = {"lon_offset_frozen_routeA": off_frozen, "lon_offset_by_profile_healthy": off_prof, "runs": {}, "groups": {}}
    feats = {}
    for key, r in runs.items():
        prof = r["profile"]
        off = off_prof.get(prof, {}).get("median", off_frozen) if prof != "std" else off_frozen
        E, why = extract(HERE / r["run_dir"], off)
        if E is None:
            out["runs"][key] = {"status": why}; continue
        E["lon_offset_used"] = off
        if prof != "std":   # longitudinal metrics when labelled with the frozen routeA offset, alongside
            E2, _ = extract(HERE / r["run_dir"], off_frozen)
            E["Y_frozen_offset"] = E2["Y"] if E2 is not None else None
        feats[key] = E
        info = {"family": FAMILY.get(r["scenario_id"], "?"), "profile": prof, "covtype": r["covtype"], "probe": r["probe"],
                "n_anchors": int(len(E["X"])), "lon_offset_used": off, "duration_s": float(E["t"][-1] - E["t"][0]),
                "max_speed_mps": float(np.nanmax([float(row["carla_ego_speed"] or "nan")
                                                  for row in csv.DictReader(open(HERE / r["run_dir"] / "long_route_pilot.csv"))]))}
        # frozen model outputs
        for ax in AXES:
            ai = AXES.index(ax); y = E["Y"][:, ai]; t1, w3 = mc.T1[ax], mc.W3[ax]
            o = models[ax].outputs(E["X"])
            band = np.where(y <= t1, "normal", np.where(y <= w3, "middle", "div"))
            lab1 = (y > t1).astype(int)
            info[ax] = {"cov": round(float((y <= o["q90_conf"]).mean()), 4),
                        "cov_by_band": {b: round(float((y[band == b] <= o["q90_conf"][band == b]).mean()), 3)
                                        for b in ("normal", "middle", "div") if (band == b).sum() >= 10},
                        "n_by_band": {b: int((band == b).sum()) for b in ("normal", "middle", "div")},
                        "auroc_T1": round(mc.auroc(o["p_T1"], lab1), 4) if 0 < lab1.sum() < len(lab1) else None,
                        "FA_rate_pT1_ge_0.5": round(float((o["p_T1"][y <= t1] >= 0.5).mean()), 4) if (y <= t1).any() else None,
                        "p_T1_mean": round(float(o["p_T1"].mean()), 4), "p_3W_mean": round(float(o["p_3W"].mean()), 4),
                        "p_T1_div_mean": round(float(o["p_T1"][band == "div"].mean()), 4) if (band == "div").any() else None,
                        "p_3W_div_mean": round(float(o["p_3W"][band == "div"].mean()), 4) if (band == "div").any() else None,
                        "q90_conf_div_median": round(float(np.median(o["q90_conf"][band == "div"])), 3) if (band == "div").any() else None,
                        "y_max": round(float(y.max()), 3), "viol": round(float(o["viol"]), 6)}
            E[f"out_{ax}"] = o
            if ax == "lon" and E.get("Y_frozen_offset") is not None:
                y2 = E["Y_frozen_offset"][:, ai]
                info[ax]["with_frozen_offset"] = {"cov": round(float((y2 <= o["q90_conf"]).mean()), 4),
                                                  "FA_rate_pT1_ge_0.5": round(float((o["p_T1"][y2 <= t1] >= 0.5).mean()), 4) if (y2 <= t1).any() else None,
                                                  "auroc_T1": round(mc.auroc(o["p_T1"], (y2 > t1).astype(int)), 4) if 0 < (y2 > t1).sum() < len(y2) else None}
        # group A: Laplace covariance baseline (same runs)
        if r["covtype"].startswith("laplace"):
            C = read_cov(HERE / r["run_dir"])
            info["exe_time_ms"] = {"mean": float(np.nanmean(C["exe_ms"])), "p99": float(np.nanpercentile(C["exe_ms"], 99))} if np.isfinite(C["exe_ms"]).any() else None
            for ax, sk in (("lon", "sig_lon"), ("lat", "sig_lat"), ("yaw", "sig_yaw")):
                ai = AXES.index(ax); y = E["Y"][:, ai]; t1, w3 = mc.T1[ax], mc.W3[ax]
                # latest covariance at the anchor time (last message before the anchor) -> 3-sigma bound (label = 1 s maximum error)
                j = np.searchsorted(C["t"], E["t_anchor"], side="right") - 1
                sig = C[sk][np.clip(j, 0, len(C["t"]) - 1)]
                b3 = 3.0 * sig
                band = np.where(y <= t1, "normal", np.where(y <= w3, "middle", "div"))
                lab1 = (y > t1).astype(int)
                info[ax]["laplace3s"] = {
                    "sigma_median": round(float(np.median(sig)), 4), "sigma_max": round(float(sig.max()), 4),
                    "frac_sigma_above_floor": round(float((sig > np.sqrt(0.0225) + 1e-6).mean()), 4) if ax != "yaw" else 0.0,
                    "cov": round(float((y <= b3).mean()), 4),
                    "cov_by_band": {b: round(float((y[band == b] <= b3[band == b]).mean()), 3)
                                    for b in ("normal", "middle", "div") if (band == b).sum() >= 10},
                    "cov_fixed3s": round(float((y <= FIXED_3S[ax]).mean()), 4),
                    "cov_fixed3s_by_band": {b: round(float((y[band == b] <= FIXED_3S[ax]).mean()), 3)
                                            for b in ("normal", "middle", "div") if (band == b).sum() >= 10},
                    "spearman_sigma_vs_err": round(mc.spearman(sig, y), 4),
                    "auroc_T1_sigma_as_score": round(mc.auroc(sig, lab1), 4) if 0 < lab1.sum() < len(lab1) else None,
                    "mean_width_asinh": round(float(np.arcsinh(b3).mean()), 4),
                    "MELD_mean_width_asinh": round(float(np.arcsinh(E[f"out_{ax}"]["q90_conf"]).mean()), 4)}
        out["runs"][key] = info
    # 2) group summaries
    def pool(keys, ax, field, sub=None):
        ys, bs = [], []
        for k in keys:
            E = feats.get(k)
            if E is None:
                continue
            ai = AXES.index(ax); ys.append(E["Y"][:, ai]); bs.append(E[f"out_{ax}"]["q90_conf"] if field == "meld" else None)
        return ys, bs
    groups = {"A_laplace": [k for k, r in runs.items() if r["covtype"] == "laplace_cal"],
              "A0_measure": [k for k, r in runs.items() if r["covtype"] in ("laplace", "laplace_raw")],
              "B_std_healthy": [k for k, r in runs.items() if r["scenario_id"] == "healthy" and r["profile"] == "std" and r["covtype"] == "fixed"],
              "B_fast_healthy": [k for k, r in runs.items() if r["scenario_id"] == "healthy" and r["profile"] == "fast" and r["covtype"] == "fixed"],
              "B_aggr_healthy": [k for k, r in runs.items() if r["scenario_id"] == "healthy" and r["profile"] == "aggr" and r["covtype"] == "fixed"],
              "B_fast_degraded": [k for k, r in runs.items() if r["scenario_id"] != "healthy" and r["profile"] == "fast"],
              "B_aggr_degraded": [k for k, r in runs.items() if r["scenario_id"] != "healthy" and r["profile"] == "aggr"],
              "C_stronger": [k for k, r in runs.items() if r["scenario_id"].startswith("st_")]}
    for g, keys in groups.items():
        keys = [k for k in keys if k in feats]
        if not keys:
            continue
        G = {"n_runs": len(keys), "runs": keys}
        for ax in AXES:
            ai = AXES.index(ax); t1, w3 = mc.T1[ax], mc.W3[ax]
            y = np.concatenate([feats[k]["Y"][:, ai] for k in keys])
            q = np.concatenate([feats[k][f"out_{ax}"]["q90_conf"] for k in keys])
            p1 = np.concatenate([feats[k][f"out_{ax}"]["p_T1"] for k in keys])
            band = np.where(y <= t1, "normal", np.where(y <= w3, "middle", "div"))
            lab1 = (y > t1).astype(int)
            G[ax] = {"n": int(len(y)), "cov": round(float((y <= q).mean()), 4),
                     "cov_by_band": {b: round(float((y[band == b] <= q[band == b]).mean()), 3) for b in ("normal", "middle", "div") if (band == b).sum() >= 20},
                     "n_by_band": {b: int((band == b).sum()) for b in ("normal", "middle", "div")},
                     "auroc_T1": round(mc.auroc(p1, lab1), 4) if 0 < lab1.sum() < len(lab1) else None,
                     "FA_rate_pT1_ge_0.5": round(float((p1[y <= t1] >= 0.5).mean()), 4) if (y <= t1).any() else None,
                     "FA_rate_pT1_ge_0.3": round(float((p1[y <= t1] >= 0.3).mean()), 4) if (y <= t1).any() else None}
            if g == "A_laplace":
                for fld in ("cov", "cov_by_band", "cov_fixed3s", "cov_fixed3s_by_band"):
                    pass
                ys = y; sig3 = np.concatenate([3.0 * (read_cov(HERE / runs[k]["run_dir"])[{"lon": "sig_lon", "lat": "sig_lat", "yaw": "sig_yaw"}[ax]][
                    np.clip(np.searchsorted(read_cov(HERE / runs[k]["run_dir"])["t"], feats[k]["t_anchor"], side="right") - 1, 0, None)]) for k in keys])
                G[ax]["laplace3s"] = {"cov": round(float((ys <= sig3).mean()), 4),
                                      "cov_by_band": {b: round(float((ys[band == b] <= sig3[band == b]).mean()), 3) for b in ("normal", "middle", "div") if (band == b).sum() >= 20},
                                      "cov_fixed3s": round(float((ys <= FIXED_3S[ax]).mean()), 4),
                                      "cov_fixed3s_by_band": {b: round(float((ys[band == b] <= FIXED_3S[ax]).mean()), 3) for b in ("normal", "middle", "div") if (band == b).sum() >= 20},
                                      "spearman_sigma_vs_err": round(mc.spearman(sig3, ys), 4),
                                      "auroc_T1_sigma_as_score": round(mc.auroc(sig3, lab1), 4) if 0 < lab1.sum() < len(lab1) else None,
                                      "frac_above_floor": round(float((sig3 > 3 * np.sqrt(0.0225) + 1e-6).mean()), 4) if ax != "yaw" else 0.0,
                                      "mean_width_asinh": round(float(np.arcsinh(sig3).mean()), 4), "MELD_mean_width_asinh": round(float(np.arcsinh(q).mean()), 4)}
        out["groups"][g] = G
    np.savez_compressed(NPZ, **{f"{k}__X": v["X"] for k, v in feats.items()}, **{f"{k}__Y": v["Y"] for k, v in feats.items()},
                        **{f"{k}__t": v["t_anchor"] for k, v in feats.items()})
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["groups"], indent=1)[:6000])
    print("->", OUT)


if __name__ == "__main__":
    main()
