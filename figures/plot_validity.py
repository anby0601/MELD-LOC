#!/usr/bin/env python3
"""Two figures computed from the shipped artifacts (read-only).

Figure 1, structural validity: raw curves before any repair - MELD (after isotonic
  calibration, before the cumulative-minimum projection) / Q-multi (before sorting) / K-indep (raw).
  Anchor selection (fixed in advance): axis = the one whose Q-multi anchor crossing rate is the
  median of the three (yaw); among its crossing anchors, the one whose number of crossing pairs is
  closest to the median (lowest index on ties). The violation-rate panel is recomputed per adjacent
  pair for every method (MELD = adjacent grid points, K-indep = adjacent thresholds, Q-multi = adjacent quantiles).
Figure 2, trajectory with the bound: run ho_rc_early12_r51 (holdout; the longest continuous
  longitudinal mid-band stay, 15.2 s). The error transform is verified in code to equal the label
  pipeline of build_features.py; plotted after the LON_OFFSET (-1.6513 m) correction.
Usage: plot_validity.py --lang en|ko   (ko = Korean labels for presentations)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "models/features"))
sys.path.insert(0, str(HERE / "models/meldloc"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import build_features as hc  # noqa: E402
import meldloc_core as mc  # noqa: E402

OUTD = HERE / "figures/out"
CACHE = HERE / "results/features/development_features.npz"
RUN2 = HERE / "data/logs/holdout/ho_rc_early12_r51_a1"
HONPZ = HERE / "results/features/holdout_features.npz"
LON_OFF = hc.LON_OFFSETS["routeA"]
QL = (.1, .2, .3, .4, .5, .6, .7, .8, .9)

# colour-blind-safe palette (Okabe-Ito) plus distinct line styles
C = {"meld": "#0072B2", "qm": "#E69F00", "ki": "#009E73", "bad": "#D55E00",
     "gt": "#000000", "ndt": "#0072B2", "fix": "#999999", "band": "#56B4E9"}

TXT = {
 "ko": dict(p_exceed="P(오차 > t)", thr_deg="오차 임계 t [°] (asinh 축)",
            meld="MELD-Loc", qm="Q-multi", ki="K-indep",
            viol="단조 위반", zero="0", rate="인접 쌍 위반율 [%]",
            mono0="위반 0", violn="단조 위반",
            axes=["종방향", "횡방향", "요"], axs=["종", "횡", "요"],
            gt="GT 궤적", ndt="NDT 추정",
            shade="종방향 중간 열화 구간", x2="진행거리 [m]",
            y2="종방향 오차 (오프셋 보정 후) [m]", y2c="|종방향 오차| [m]",
            fix="고정 3σ (±0.45 m)",
            band="MELD q90 상한 (conformal)", err="실제 오차",
            exceed="고정 밴드 초과", note="종축 asinh 스케일 (클리핑 없음)",
            anno="고정 3σ는 놓치고\nMELD 상한은 덮음",
            anno2="정상 회복 시 상한 수축", xy="지도 좌표 [m]"),
 "en": dict(p_exceed="P(error > t)", thr_deg="error threshold t [deg] (asinh scale)",
            meld="MELD-Loc", qm="Q-multi",
            ki="K-indep", viol="monotonicity violation", zero="0",
            mono0="0 violations",
            violn="violations",
            rate="adjacent-pair violation rate [%]",
            axes=["longitudinal", "lateral", "yaw"],
            axs=["lon", "lat", "yaw"], gt="GT trajectory",
            ndt="NDT estimate", shade="longitudinal mid-degradation",
            x2="route progress [m]",
            y2="longitudinal error (offset-corrected) [m]",
            y2c="|longitudinal error| [m]",
            fix="fixed 3-sigma (±0.45 m)", band="MELD q90 bound (conformal)",
            err="actual error", exceed="exceeds fixed band",
            note="asinh-scaled y-axis (no clipping)",
            anno="fixed 3σ misses,\nMELD bound covers",
            anno2="bound contracts on recovery", xy="map coord. [m]"),
}


def setup_font(lang):
    if lang == "ko":
        matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
    else:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 9


def savefig(fig, stem):
    OUTD.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUTD / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1(lang, T):
    import lightgbm as lgb
    Z = np.load(CACHE, allow_pickle=True)
    X = Z["X"].astype(float); Y = Z["Y"].astype(float)
    AXN = list(Z["axes"].astype(str)); split = Z["split"].astype(str)
    tr, va = split == "train", split == "validation"
    gbdt = mc.GBDT_DEFAULT
    rates = {}          # adjacent-pair violation rates, same unit for every method
    aq_yaw = None
    for ax in ("lon", "lat", "yaw"):
        y = Y[:, AXN.index(ax)]
        M = mc.MeldLoc.load(HERE / f"models/frozen/meldloc_{ax}.joblib")
        at = float(np.arcsinh(M.t_max)); tg9 = np.sinh(np.linspace(0, at, 9))
        S_pre = M._eval_S(X[va], M.t_grid, M.lam, True)          # before the cumulative-minimum projection
        AQ = np.stack([lgb.LGBMRegressor(objective="quantile", alpha=q, **gbdt
                       ).fit(X[tr], np.arcsinh(y[tr])).predict(X[va])
                       for q in QL], axis=1)
        S_ind = np.stack([lgb.LGBMClassifier(is_unbalance=True, **gbdt).fit(
            X[tr], (y[tr] > tv).astype(int)).predict_proba(X[va])[:, 1]
            for tv in tg9], axis=1)
        rates[ax] = (float((np.diff(S_pre, axis=1) > 1e-6).mean()),
                     float((np.diff(S_ind, axis=1) > 1e-6).mean()),
                     float((np.diff(AQ, axis=1) < 0).mean()))
        if ax == "yaw":
            aq_yaw, sind_yaw, spre_yaw, M_yaw, tg9_yaw = AQ, S_ind, S_pre, M, tg9
    # anchor selection: (1) yaw axis, (2) anchors whose number of crossing pairs equals the median
    # (many ties), (3) among those, the anchor whose realized yaw error lies in the mid band (T1, 3W]
    # closest to the band centre in asinh - a stated tie-break that avoids degenerate benign curves.
    yv = Y[va][:, AXN.index("yaw")]
    ncross = (np.diff(aq_yaw, axis=1) < 0).sum(1)
    with_c = np.where(ncross >= 1)[0]
    med = float(np.median(ncross[with_c]))
    cand = with_c[np.abs(ncross[with_c] - med) == np.min(np.abs(ncross[with_c] - med))]
    t1y, w3y = mc.T1["yaw"], mc.W3["yaw"]
    midc = cand[(yv[cand] > t1y) & (yv[cand] <= w3y)]
    pool = midc if len(midc) else cand
    ctr = 0.5 * (np.arcsinh(t1y) + np.arcsinh(w3y))
    d = np.abs(np.arcsinh(yv[pool]) - ctr)
    sel = pool[np.lexsort((pool, d))][0]
    k = int(ncross[sel])
    print(f"[fig1] selected anchor idx={sel} (yaw, {k} crossing pairs; median {med:.0f}; "
          f"{len(cand)} tied, {len(midc)} in the mid band, realized |yaw error| "
          f"{yv[sel]:.2f}°)")
    print(f"[fig1] adjacent-pair violation rates (MELD/K-indep/Q-multi): " +
          str({a: tuple(round(v*100,1) for v in r) for a, r in rates.items()}))

    # layout: three mini panels (same anchor, same input) with violations as red arrows, plus the
    # bar panel; 1x4 full-width arrangement at 1:1 scale (figsize 7.2 in ~ \textwidth, no font scaling).
    ash = np.arcsinh
    tgrid = np.asarray(M_yaw.t_grid)
    fig = plt.figure(figsize=(7.2, 2.0))
    # odd columns = spacing so titles/labels do not touch the neighbouring panel
    gs = fig.add_gridspec(1, 7,
                          width_ratios=[1, 0.07, 1, 0.07, 1, 0.13, 1.3],
                          wspace=0.20)
    axm = fig.add_subplot(gs[0, 0])
    axk = fig.add_subplot(gs[0, 2], sharex=axm, sharey=axm)
    axq = fig.add_subplot(gs[0, 4], sharex=axm, sharey=axm)
    a2 = fig.add_subplot(gs[0, 6])

    def mini(ax, xs, ys, col, title, viol_pairs, ls="-", marker=None):
        ax.plot(xs, ys, color=col, lw=2.0, ls=ls, marker=marker, ms=3.5,
                zorder=3)
        for jj in viol_pairs:  # increasing segment = red arrow (the definition of a violation)
            ax.annotate("", xy=(xs[jj + 1], ys[jj + 1]),
                        xytext=(xs[jj], ys[jj]), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=C["bad"],
                                        lw=1.8, shrinkA=0, shrinkB=0))
        ax.set_title(title, loc="left", fontsize=7, pad=2,
                     color=col if not viol_pairs else "black")
        ax.set_ylim(-0.07, 1.07); ax.set_yticks([0, 1])
        ax.tick_params(labelsize=7)

    xg = ash(tgrid)
    mini(axm, xg, spre_yaw[sel], C["meld"],
         f"{T['meld']}\n{T['mono0']}", [])
    vk = list(np.where(np.diff(sind_yaw[sel]) > 1e-6)[0])
    mini(axk, ash(tg9_yaw), sind_yaw[sel], C["ki"],
         f"{T['ki']}\n{len(vk)} {T['violn']}", vk, ls=":", marker="s")
    qv = np.sinh(aq_yaw[sel]); pv = 1 - np.arange(1, 10) / 10.0
    xq = ash(np.maximum(qv, 0))
    vq = list(np.where(np.diff(aq_yaw[sel]) < 0)[0])
    mini(axq, xq, pv, C["qm"], f"{T['qm']}\n{len(vq)} {T['violn']}", vq,
         ls="--", marker="o")
    ticks = [0, 1, 10, 100]
    for ax in (axm, axk, axq):
        ax.set_xticks([ash(t) for t in ticks])
        ax.set_xticklabels([str(t) for t in ticks])
    for ax in (axk, axq):
        plt.setp(ax.get_yticklabels(), visible=False)
    axk.set_xlabel(T["thr_deg"], fontsize=8)
    axm.set_ylabel(T["p_exceed"], fontsize=8)
    # right: adjacent-pair violation rates
    labels = T["axs"]; yy = np.arange(3)
    h = 0.25
    # method names without pre-repair/pre-sort qualifiers (the caption states "raw"); MELD bar labelled "0 violations"
    for i, (m, key, col) in enumerate(
            [(T["meld"], 0, C["meld"]), (T["ki"], 1, C["ki"]),
             (T["qm"], 2, C["qm"])]):
        vals = [rates[a][key] * 100 for a in ("lon", "lat", "yaw")]
        if key == 0:
            # MELD = 0: draw a zero-length bar outline so the three methods stay aligned
            for j in range(3):
                yb = yy[j] + (i - 1) * h
                a2.plot([0, 0], [yb - h / 2, yb + h / 2], color=col, lw=2.2,
                        solid_capstyle="butt")
                a2.text(1.0, yb, T["zero"], va="center", fontsize=8, color=col)
        else:
            a2.barh(yy + (i - 1) * h, vals, height=h, color=col,
                    label=m, edgecolor="none")
    a2.set_yticks(yy); a2.set_yticklabels(labels, fontsize=8)
    a2.set_xlabel(T["rate"], fontsize=8)
    a2.xaxis.set_label_coords(0.42, -0.155)  # shifted left to avoid clipping at the right edge
    a2.tick_params(labelsize=7)
    a2.invert_yaxis()
    a2.legend(frameon=False, fontsize=7, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0)
    fig.tight_layout()
    savefig(fig, f"validity_{lang}")
    return sel, k, rates


def fig2(lang, T):
    rr = list(csv.DictReader(open(RUN2 / "long_route_pilot.csv")))
    g = lambda k: np.array([float(r[k]) for r in rr])
    t, pr = g("timestamp"), g("route_progress_m")
    gx, gy, gyaw = g("gt_x"), g("gt_y"), g("gt_yaw")
    nx, ny = g("ndt_x"), g("ndt_y")
    lon_raw = g("raw_longitudinal_residual")
    # check that the error transform equals the label pipeline (gt_yaw rotation of the logger)
    yawr = np.radians(gyaw)
    lon_re = np.cos(yawr) * (nx - gx) + np.sin(yawr) * (ny - gy)
    ok = np.nanmax(np.abs(lon_re - lon_raw)) < 1e-6
    print(f"[fig2] transform identity: {'PASS' if ok else 'FAIL'} "
          f"(max|Δ|={np.nanmax(np.abs(lon_re-lon_raw)):.2e})")
    e = lon_raw - LON_OFF                       # signed error after the offset correction
    mid = (np.abs(e) > mc.T1["lon"]) & (np.abs(e) <= mc.W3["lon"])
    # MELD q90 bound: the anchors of this run from the holdout features
    Z = np.load(HONPZ, allow_pickle=True)
    sel = Z["scenario"].astype(str) == "ho_rc_early12_r51"
    Xa, ta = Z["X"].astype(float)[sel], Z["t_anchor"].astype(float)[sel]
    M = mc.MeldLoc.load(HERE / "models/frozen/meldloc_lon.joblib")
    q90 = M.outputs(Xa)["q90_conf"]
    pr_a = np.interp(ta, t, pr)
    print(f"[fig2] {len(ta)} anchors, q90 range {q90.min():.2f}-{q90.max():.1f} m")

    # mid-band degradation segments (shared by the three panels)
    tt = t[mid]
    brk = np.where(np.diff(tt) > 0.5)[0]
    segs = [s for s in np.split(tt, brk + 1) if len(s) > 1]
    # zoom window: the onset-and-recovery segment containing the first mid-band entry (the longest
    # stay lies late in the collapse where the local q90 is 52-156 m, so it is not the zoom target).
    ZW = (10.0, 72.0)                            # route progress [m]
    print(f"[fig2] zoom window pr={ZW[0]}-{ZW[1]} m (contains the first mid-band entry at pr="
          f"{pr[mid][0]:.1f} m)")

    fig = plt.figure(figsize=(7.2, 9.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.62, 1.0], hspace=0.34)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1])
    a3 = fig.add_subplot(gs[2])

    # (a) trajectory
    a1.plot(gx, gy, color=C["gt"], lw=1.4, label=T["gt"])
    a1.plot(nx, ny, color=C["ndt"], ls="--", lw=0.9, alpha=0.8, label=T["ndt"])
    # highlight the mid-band segments; NaN breaks avoid straight lines across gaps
    gxm = np.where(mid, gx, np.nan); gym = np.where(mid, gy, np.nan)
    a1.plot(gxm, gym, color=C["bad"], lw=5.5, alpha=0.9, label=T["shade"],
            solid_capstyle="round", zorder=4)
    a1.set_aspect("equal"); a1.set_xlabel(T["xy"]); a1.set_ylabel(T["xy"])
    a1.legend(frameon=False, fontsize=8, loc="lower right")

    # (b) error and bound along the whole route (asinh axis; shows the divergence magnitude honestly)
    F = np.arcsinh

    def band_xy(px, qy):
        # several anchors per progress bin (0.5 s anchors vs 2 m grid) -> maximum q90
        ux = np.unique(px)
        return ux, np.array([qy[px == u].max() for u in ux])

    for s in segs:
        p0, p1 = np.interp([s[0], s[-1]], t, pr)
        a2.axvspan(p0, p1, color=C["bad"], alpha=0.12, lw=0)
    a2.axhspan(-F(0.45), F(0.45), color=C["fix"], alpha=0.55, lw=0)
    ub, qb = band_xy(pr_a, q90)
    a2.fill_between(ub, -F(qb), F(qb), color=C["band"], alpha=0.35, lw=0)
    a2.plot(pr, F(e), color=C["gt"], lw=0.9)
    yt = [-300, -8.17, -2.5, 0, 2.5, 8.17, 300]
    a2.set_yticks([F(v) for v in yt])
    a2.set_yticklabels(["-300", "-3W", "-T1", "0", "T1", "3W", "300"],
                       fontsize=8)
    # zoom-window marker for (c)
    a2.axvspan(*ZW, fill=False, edgecolor="0.15", lw=1.1, ls="--", zorder=5)
    a2.text(0.5 * (ZW[0] + ZW[1]), F(430), "(c)", ha="center", fontsize=9,
            weight="bold")
    a2.set_ylim(-F(600), F(600))
    a2.text(0.985, 0.04, T["note"], transform=a2.transAxes, ha="right",
            fontsize=7.5, color="0.35")
    a2.set_xlabel(T["x2"]); a2.set_ylabel(T["y2"], fontsize=8)

    # (c) zoom on the degradation onset, linear axis
    w = (pr >= ZW[0]) & (pr <= ZW[1])
    wa = (pr_a >= ZW[0]) & (pr_a <= ZW[1])
    for s in segs:
        p0, p1 = np.interp([s[0], s[-1]], t, pr)
        if p1 >= ZW[0] and p0 <= ZW[1]:
            a3.axvspan(max(p0, ZW[0]), min(p1, ZW[1]), color=C["bad"],
                       alpha=0.10, lw=0)
    # magnitude axis |e| (the bound is a q90 of |error|)
    a3.axhspan(0, 0.45, color=C["fix"], alpha=0.55, lw=0, label=T["fix"])
    uc, qc = band_xy(pr_a[wa], q90[wa])
    a3.fill_between(uc, 0, qc, color=C["band"], alpha=0.35, lw=0,
                    label=T["band"])
    a3.plot(uc, qc, color=C["meld"], lw=1.2)
    ae = np.abs(e)
    a3.plot(pr[w], ae[w], color=C["gt"], lw=1.2, label=T["err"])
    bw = w & (ae > 0.45)
    a3.plot(pr[bw], ae[bw], color=C["bad"], ls="none", marker=".", ms=3.0,
            label=T["exceed"])
    a3.set_xlim(*ZW); a3.set_ylim(0, 12.5)
    a3.set_yticks([0, 0.45, 2.5, 8.17])
    a3.set_yticklabels(["0", ".45", "T1", "3W"], fontsize=8)
    a3.set_xlabel(T["x2"]); a3.set_ylabel(T["y2c"])
    # annotation 1: middle of the first mid-band stay
    s0 = segs[0]
    i = int(np.argmin(np.abs(t - s0[len(s0) // 2])))
    a3.annotate(T["anno"], xy=(pr[i], ae[i]), xytext=(ZW[0] + 1.5, 9.8),
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                          alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=C["bad"]))
    # annotation 2: the bound shrinks on recovery
    ir = int(np.argmin(np.abs(pr_a - 57.0)))
    a3.annotate(T["anno2"], xy=(pr_a[ir], q90[ir] + 0.15),
                xytext=(49.0, 5.2), fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                          alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=C["meld"]))
    a3.legend(fontsize=7.5, loc="upper right", ncols=2, framealpha=0.9,
              edgecolor="none")

    for ax, lab in ((a1, "(a)"), (a2, "(b)"), (a3, "(c)")):
        ax.text(-0.06, 1.04, lab, transform=ax.transAxes, fontsize=10,
                weight="bold", va="bottom")
    savefig(fig, f"trajectory_{lang}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ko", "en"], required=True)
    a = ap.parse_args()
    T = TXT[a.lang]
    setup_font(a.lang)
    fig1(a.lang, T)
    if RUN2.exists():
        fig2(a.lang, T)
    else:
        print("fig2 skipped: archived run", RUN2, "is not present (needs the data archive)")
    print("done:", OUTD)


if __name__ == "__main__":
    main()
