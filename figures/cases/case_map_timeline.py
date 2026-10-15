#!/usr/bin/env python3
"""Map and timeline of one run: ho_ls08_r52 (lateral_shear 0.8, holdout).

(a) global map view + zoom boxes  (e) lateral error vs bounds timeline
(b)(c)(d) zooms: normal / degraded / re-converged
All content from frozen logs + frozen model outputs. No synthetic elements.
"""
import os
import sys
from pathlib import Path

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parents[2]
BASE = str(HERE)
CACHE = HERE / 'figures/out/_hero_cache'
CACHE.mkdir(parents=True, exist_ok=True)
SCRATCH = str(CACHE)
RUN = 'ho_ls08_r52'
OFF = -1.6512603049722019       # frozen LON_OFFSETS['routeA'] (model_freeze_manifest.json)
BL = 0.45                       # Autoware default sigma_xy 0.15 m x scale 3
PCD = os.path.join(os.environ.get('AUTOWARE_MAP_DIR', os.path.expanduser('~/autoware_map')), 'Town01/pointcloud_map.pcd')


def _ensure_q90():
    """Per-anchor q90 from the FROZEN models on the sealed holdout features (read-only)."""
    out = CACHE / 'holdout_q90.npz'
    if out.exists():
        return
    sys.path.insert(0, str(HERE / 'models/meldloc'))
    sys.path.insert(0, str(HERE / 'models/features'))
    import meldloc_core as mc
    z = np.load(HERE / 'results/features/holdout_features.npz', allow_pickle=True)
    arrs = {}
    for ax in ('lon', 'lat', 'yaw'):
        M = mc.MeldLoc.load(HERE / f'models/frozen/meldloc_{ax}.joblib')
        arrs[f'q90_{ax}'] = M.outputs(z['X'])['q90_conf']
    np.savez(out, scenario=z['scenario'], t=z['t_anchor'], Y=z['Y'], **arrs)


def _ensure_map():
    """Decompress the Town01 PCD (binary_compressed) and crop to the route bbox."""
    out = CACHE / 'town01_crop.npz'
    if out.exists():
        return
    import lzf, struct
    raw = open(PCD, 'rb').read()
    h = raw.index(b'DATA binary_compressed\n') + len(b'DATA binary_compressed\n')
    csz, usz = struct.unpack('<II', raw[h:h+8])
    a = np.frombuffer(lzf.decompress(raw[h+8:h+8+csz], usz), dtype='<f4')
    n = usz // 12
    x, y, zc = a[:n], a[n:2*n], a[2*n:3*n]
    m = (x > 60) & (x < 410) & (y > -230) & (y < 30)
    np.savez(out, x=x[m], y=y[m], z=zc[m])


_ensure_q90()
_ensure_map()

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                     'font.size': 7, 'axes.labelsize': 7.5, 'axes.linewidth': 0.6,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})

C_GT, C_NDT, C_BL, C_Q = 'black', '#c8102e', '#1f5fbf', '#e87d0d'

# ---------- data ----------
class _Run:                                   # thin numpy record wrapper
    pass


d = _Run()
cols = ('timestamp', 'gt_x', 'gt_y', 'gt_yaw', 'ndt_x', 'ndt_y',
        'raw_lateral_residual')
rows = []
with open(f'{BASE}/data/logs/holdout/{RUN}_a1/long_route_pilot.csv') as fh:
    for r in csv.DictReader(fh):
        if r['phase'] != 'route':
            continue
        try:
            rows.append([float(r[c]) for c in cols])
        except ValueError:                     # NaN/empty gt or ndt sample
            continue
arr = np.array(rows)
arr = arr[np.all(np.isfinite(arr), axis=1)]
for i, c in enumerate(cols):
    setattr(d, c, arr[:, i])
d.gt_yaw_deg = d.gt_yaw
th = np.radians(d.gt_yaw_deg)
ux, uy = np.cos(th), np.sin(th)          # GT heading (frame of the lateral residual)
nx, ny = -uy, ux                          # left normal
d.cx = d.ndt_x - OFF * ux                # remove frozen route-constant lon offset
d.cy = d.ndt_y - OFF * uy
d.elat = np.abs(d.raw_lateral_residual)
t = d.timestamp

z = np.load(f'{SCRATCH}/holdout_q90.npz', allow_pickle=True)
m = z['scenario'] == RUN
o = np.argsort(z['t'][m])
ta, qa = z['t'][m][o], z['q90_lat'][m][o]
q = np.interp(t, ta, qa)                  # q90 on the 10 Hz timeline

mp = np.load(f'{SCRATCH}/town01_crop.npz')
mx, my, mz = mp['x'], mp['y'], mp['z']

# ---------- layout ----------
fig = plt.figure(figsize=(7.16, 3.55))
gs = fig.add_gridspec(2, 12, height_ratios=[1.12, 1.0],
                      left=0.045, right=0.995, top=0.975, bottom=0.10,
                      hspace=0.28, wspace=1.2)
axA = fig.add_subplot(gs[0, 0:5])
axE = fig.add_subplot(gs[0, 5:12])
axB = fig.add_subplot(gs[1, 0:4])
axC = fig.add_subplot(gs[1, 4:8])
axD = fig.add_subplot(gs[1, 8:12])

# ---------- (a) global ----------
sub = np.random.default_rng(0).choice(len(mx), 500_000, replace=False)
ground = mz[sub] < 0.5
axA.scatter(mx[sub][ground], my[sub][ground], s=0.05, c='0.90', lw=0, rasterized=True)
axA.scatter(mx[sub][~ground], my[sub][~ground], s=0.05, c='0.74', lw=0, rasterized=True)
# degraded-segment halo: where the realized lateral error exceeds the
# deployed fixed bound (data-defined, matches the (e) exceedance episodes)
deg = d.elat > BL
axA.plot(np.where(deg, d.gt_x, np.nan), np.where(deg, d.gt_y, np.nan),
         color=C_Q, lw=3.4, alpha=0.45, solid_capstyle='round', zorder=2)
axA.plot(d.gt_x, d.gt_y, color=C_GT, lw=0.9)
axA.plot(d.cx, d.cy, color=C_NDT, lw=0.7)
axA.plot(d.gt_x[0], d.gt_y[0], marker='o', ms=3, color=C_GT, mfc='white', mew=0.8)
axA.set_aspect('equal'); axA.set_xlim(62, 408); axA.set_ylim(-225, 28)
axA.set_xticks([]); axA.set_yticks([])
for sp in axA.spines.values():
    sp.set_color('0.55')
# 50 m scale bar
axA.plot([80, 130], [-212, -212], color='k', lw=1.2)
axA.text(105, -206, '50 m', ha='center', va='bottom', fontsize=6)
axA.text(0.02, 0.965, '(a)', transform=axA.transAxes, fontsize=8, va='top', weight='bold')

# ---------- zooms ----------
ZOOMS = [(axB, '(b)', 'normal',        (93.0, 100.5)),
         (axC, '(c)', 'degraded',      (140.0, 149.0)),
         (axD, '(d)', 're-converged',  (228.0, 235.5))]
HW, HH = 13.0, 9.0                        # zoom half-extent [m], aspect 1.444

for ax, tag, lab, (t0, t1) in ZOOMS:
    wc = (t >= t0) & (t <= t1)
    cxc = 0.5 * (d.gt_x[wc].min() + d.gt_x[wc].max())
    cyc = 0.5 * (d.gt_y[wc].min() + d.gt_y[wc].max())
    box = (mx > cxc-HW-2) & (mx < cxc+HW+2) & (my > cyc-HH-2) & (my < cyc+HH+2)
    g = mz[box] < 0.5
    ax.scatter(mx[box][g], my[box][g], s=0.12, c='0.90', lw=0, rasterized=True)
    ax.scatter(mx[box][~g], my[box][~g], s=0.12, c='0.72', lw=0, rasterized=True)

    w = (t >= t0 - 3) & (t <= t1 + 3)     # margin so corridors span the panel
    cx_, cy_, nx_, ny_, q_ = d.cx[w], d.cy[w], nx[w], ny[w], q[w]
    # MELD-Loc q90 corridor (translucent fill)
    lx, ly = cx_ + q_*nx_, cy_ + q_*ny_
    rx, ry = cx_ - q_*nx_, cy_ - q_*ny_
    ax.fill(np.r_[lx, rx[::-1]], np.r_[ly, ry[::-1]], color=C_Q, alpha=0.28, lw=0)
    ax.plot(lx, ly, color=C_Q, lw=0.6); ax.plot(rx, ry, color=C_Q, lw=0.6)
    # fixed 3-sigma corridor (dashed boundaries)
    ax.plot(cx_ + BL*nx_, cy_ + BL*ny_, color=C_BL, lw=0.8, ls=(0, (3.2, 1.8)))
    ax.plot(cx_ - BL*nx_, cy_ - BL*ny_, color=C_BL, lw=0.8, ls=(0, (3.2, 1.8)))
    # trajectories
    ax.plot(cx_, cy_, color=C_NDT, lw=1.0)
    gx, gy = d.gt_x[w], d.gt_y[w]
    ax.plot(gx, gy, color=C_GT, lw=1.0)

    ax.set_aspect('equal')
    ax.set_xlim(cxc-HW, cxc+HW); ax.set_ylim(cyc-HH, cyc+HH)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('0.55')
    ax.plot([cxc+HW-6.5, cxc+HW-1.5], [cyc-HH+1.6, cyc-HH+1.6], color='k', lw=1.1)
    ax.text(cxc+HW-4.0, cyc-HH+2.3, '5 m', ha='center', va='bottom', fontsize=6)
    ax.text(0.03, 0.955, tag, transform=ax.transAxes, fontsize=8, va='top', weight='bold')
    ax.text(0.97, 0.955, lab, transform=ax.transAxes, fontsize=7, va='top',
            ha='right', style='italic')
    # zoom box + label on (a)
    axA.add_patch(Rectangle((cxc-HW, cyc-HH), 2*HW, 2*HH, fill=False,
                            ec='k', lw=0.7))
    axA.annotate(tag[1], (cxc, cyc), xytext=(cxc+20, cyc+16), fontsize=7,
                 arrowprops=dict(arrowstyle='-', lw=0.5))

# ---------- (c) callout: the information the fixed covariance misses ----------
x0, x1 = axC.get_xlim()
y0, y1 = axC.get_ylim()
wc = ((t >= ZOOMS[1][3][0]) & (t <= ZOOMS[1][3][1])
      & (d.gt_x > x0 + 1) & (d.gt_x < x1 - 1)
      & (d.gt_y > y0 + 1) & (d.gt_y < y1 - 1))
ic = np.nonzero(wc)[0][np.argmax(d.elat[wc])]
axC.annotate('GT outside fixed $3\\sigma$,\ninside MELD-Loc bound',
             xy=(d.gt_x[ic], d.gt_y[ic]), xycoords='data',
             xytext=(0.05, 0.09), textcoords='axes fraction',
             fontsize=7.4, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='0.65',
                       lw=0.5, alpha=0.92),
             arrowprops=dict(arrowstyle='->', lw=0.7, color='k',
                             shrinkA=1.5, shrinkB=1.5))

# ---------- (e) timeline ----------
axE.axhline(BL, color=C_BL, lw=0.9, ls=(0, (3.2, 1.8)))
axE.plot(t, d.elat, color='0.35', lw=0.45)
axE.plot(t, q, color=C_Q, lw=1.1)
for _, tag, _, (t0, t1) in ZOOMS:
    axE.axvspan(t0, t1, color='0.85', alpha=0.55, lw=0)
    axE.text((t0+t1)/2, 1.50, tag[1], ha='center', fontsize=7)
axE.set_xlim(t.min(), t.max()); axE.set_ylim(0, 1.62)
axE.set_xlabel('time [s]', labelpad=1.5)
axE.set_ylabel('lateral error [m]', labelpad=1.5)
axE.tick_params(length=2, pad=1.5)
axE.text(0.012, 0.965, '(e)', transform=axE.transAxes, fontsize=8, va='top', weight='bold')
leg = [Line2D([], [], color=C_GT, lw=1.0, label='ground truth'),
       Line2D([], [], color=C_NDT, lw=1.0, label='NDT estimate'),
       Line2D([], [], color=C_BL, lw=0.9, ls=(0, (3.2, 1.8)),
              label='fixed $3\\sigma$ bound (deployed)'),
       Patch(facecolor=C_Q, alpha=0.35, edgecolor=C_Q,
             label='MELD-Loc $q^{90}$ bound (ours)')]
axB.legend(handles=leg, loc='upper left', bbox_to_anchor=(0.015, 0.90),
           fontsize=6.0, frameon=True, framealpha=0.88, edgecolor='0.8',
           handlelength=1.5, labelspacing=0.25, borderpad=0.35)

fig.savefig(f'{SCRATCH}/case_map_timeline.pdf', dpi=400)
fig.savefig(f'{SCRATCH}/case_map_timeline.png', dpi=220)
print('saved')
