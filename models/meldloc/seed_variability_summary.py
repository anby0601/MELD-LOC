#!/usr/bin/env python3
"""Aggregate the seed-variability runs: seed*.json + the frozen model (seed 20260803) -> summary (markdown + json).

Headline metrics: raw monotonicity violations, coverage overall/normal/mid, ECE@T1, twIBS, AUROC@T1, mid-band macro Spearman
(holdout and cross-route), dev-val lambda/shift/ECE; baseline (K-indep, Q-multi) violations and twIBS per seed.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
SV = HERE / "results/seeds"
HO = json.loads((HERE / "results/evaluation/holdout_eval.json").read_text())
CR = json.loads((HERE / "results/evaluation/crossroute_eval.json").read_text())
TR = json.loads((HERE / "results/development/train_report.json").read_text())
AXES = ("lon", "lat", "yaw")


def frozen_row(ax):
    h, c = HO["axes"][ax]["all"], CR["axes"][ax]["all"]
    return {"holdout": {"monotone_violation_pre_projection": HO["axes"][ax]["monotone_violation_pre_projection"],
                        "q90_coverage": h["q90_coverage"]["MELD_conf"], "cov_by_band": h["cov_by_band"],
                        "ECE_pT1": h["ECE_pT1"], "twIBS_MELD": h["twIBS"]["MELD"],
                        "auroc_T1_MELD": h["auroc_T1"]["MELD"], "mid_macro_R": h["mid_macro_R"],
                        "twIBS_Qmulti": h["twIBS"]["Q_multi"], "twIBS_Kindep": h["twIBS"]["K_indep"]},
            "crossroute": {"monotone_violation_pre_projection": CR["axes"][ax]["monotone_violation_pre_projection"],
                           "q90_coverage": c["q90_coverage"]["MELD_conf"], "cov_by_band": c["cov_by_band"],
                           "ECE_pT1": c["ECE_pT1"], "twIBS_MELD": c["twIBS"]["MELD"],
                           "auroc_T1_MELD": c["auroc_T1"]["MELD"], "mid_macro_R": c["mid_macro_R"],
                           "twIBS_Qmulti": c["twIBS"]["Q_multi"], "twIBS_Kindep": c["twIBS"]["K_indep"]},
            "devval": {k: TR["axes"][ax][k] for k in ("lambda", "conformal_shift_asinh", "ece_calibrated",
                                                        "q90_coverage_conformal", "monotone_violation_pre_projection")}}


def main():
    seeds = {}
    for f in sorted(glob.glob(str(SV / "seed[0-9]*.json"))):
        d = json.loads(Path(f).read_text())
        if d.get("smoke"):
            continue
        seeds[d["seed"]] = d["axes"]
    rows = []
    METRICS = [("viol", lambda r: r["monotone_violation_pre_projection"]),
               ("cov", lambda r: r["q90_coverage"]),
               ("cov_normal", lambda r: r["cov_by_band"].get("normal")),
               ("cov_mid", lambda r: r["cov_by_band"].get("middle")),
               ("ECE_T1", lambda r: r["ECE_pT1"]),
               ("twIBS", lambda r: r["twIBS_MELD"]),
               ("AUROC_T1", lambda r: r["auroc_T1_MELD"]),
               ("mid_macro_R", lambda r: r["mid_macro_R"])]
    out = {"seeds_new": sorted(seeds), "frozen_seed": 20260803, "table": {}}
    md = ["# Seed variability (frozen seed 20260803 + new seeds %s)\n" % sorted(seeds)]
    for setname in ("holdout", "crossroute"):
        md.append(f"\n## {setname}\n")
        md.append("| axis | metric | frozen | new seeds mean ± SD | min – max | n |")
        md.append("|---|---|---|---|---|---|")
        for ax in AXES:
            fr = frozen_row(ax)[setname]
            for name, fn in METRICS:
                vals = [fn(seeds[s][ax][setname]) for s in sorted(seeds) if ax in seeds[s] and setname in seeds[s][ax]]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                v = np.array(vals, float)
                out["table"].setdefault(setname, {}).setdefault(ax, {})[name] = {
                    "frozen": fn(fr), "mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
                    "min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "n": int(len(v)), "values": vals}
                md.append(f"| {ax} | {name} | {fn(fr)} | {v.mean():.4f} ± {v.std(ddof=1) if len(v)>1 else 0:.4f} | {v.min():.4f} – {v.max():.4f} | {len(v)} |")
        # baseline violations and twIBS
        md.append(f"\n### baselines on {setname} (same seed)\n")
        md.append("| axis | K-indep viol | Q-multi quantile crossing | twIBS K-indep | twIBS Q-multi |")
        md.append("|---|---|---|---|---|")
        for ax in AXES:
            bl = [seeds[s][ax]["baselines"][setname] for s in sorted(seeds)
                  if ax in seeds[s] and "baselines" in seeds[s][ax]]
            if not bl:
                continue
            kv = np.array([b["K_indep"]["viol"] for b in bl]); qv = np.array([b["Q_multi"]["viol_quantile_crossing"] for b in bl])
            kt = np.array([b["K_indep"]["twIBS"] for b in bl]); qt = np.array([b["Q_multi"]["twIBS"] for b in bl])
            out["table"][setname][ax]["baselines"] = {"K_viol": kv.tolist(), "Q_cross": qv.tolist(), "K_twIBS": kt.tolist(), "Q_twIBS": qt.tolist()}
            md.append(f"| {ax} | {kv.mean():.3f} ({kv.min():.3f}–{kv.max():.3f}) | {qv.mean():.3f} ({qv.min():.3f}–{qv.max():.3f}) | "
                      f"{kt.mean():.4f} | {qt.mean():.4f} |")
    md.append("\n## dev-val fitted components\n")
    md.append("| axis | λ (frozen / seeds) | shift δ (frozen / mean ± SD) | ECE cal (frozen / mean) | cov conf (frozen / mean ± SD) |")
    md.append("|---|---|---|---|---|")
    for ax in AXES:
        fr = frozen_row(ax)["devval"]
        dv = [seeds[s][ax]["devval"] for s in sorted(seeds) if ax in seeds[s]]
        if not dv:
            continue
        lam = [d["lambda"] for d in dv]; sh = np.array([d["conformal_shift_asinh"] for d in dv])
        ec = np.array([d["ece_calibrated"] for d in dv]); cv = np.array([d["q90_coverage_conformal"] for d in dv])
        out["table"].setdefault("devval", {})[ax] = {"lambda": lam, "shift": sh.tolist(), "ece_cal": ec.tolist(), "cov_conf": cv.tolist()}
        md.append(f"| {ax} | {fr['lambda']} / {lam} | {fr['conformal_shift_asinh']:.4f} / {sh.mean():.4f} ± {sh.std(ddof=1) if len(sh)>1 else 0:.4f} | "
                  f"{fr['ece_calibrated']:.4f} / {ec.mean():.4f} | {fr['q90_coverage_conformal']:.4f} / {cv.mean():.4f} ± {cv.std(ddof=1) if len(cv)>1 else 0:.4f} |")
    (SV / "seed_variability_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (SV / "seed_variability_summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
