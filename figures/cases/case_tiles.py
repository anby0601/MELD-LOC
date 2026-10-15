#!/usr/bin/env python3
"""Per-run case figure: route map plus vehicle-centred tiles.

Panels:
  (a) route on the point-cloud map, GT trajectory coloured by realized error
  (b) three vehicle-centred tiles (normal / degraded / re-converged):
      GT vehicle at the origin (heading up), NDT-estimated vehicle outline,
      fixed 3-sigma box vs MELD-Loc q90 box around the estimate
One figure per case (one run); no time-series panel.
All content from frozen logs + frozen model outputs (holdout_q90.npz).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

HERE = Path(__file__).resolve().parents[2]
CACHE = HERE / 'figures/out/_hero_cache'
OUT = HERE / 'figures/out'
OFF = -1.6512603049722019      # frozen LON_OFFSETS['routeA']
BL = 0.45                      # deployed fixed 3-sigma (lon = lat)
CAR_L, CAR_W = 4.5, 1.9        # drawn vehicle footprint [m]

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                     'font.size': 7, 'axes.labelsize': 7.5, 'axes.linewidth': 0.6,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})
C_GT, C_NDT, C_BL, C_Q = 'black', '#c8102e', '#1f5fbf', '#e87d0d'

CASES = [
    # run, axis, label, tile instants [s from run start], tile half extents (lat, lon) [m]
    dict(run='ho_ls08_r52', axis='lat', name='lateral shear (routeA)',
         tiles=[(50.0, 'normal'), (98.0, 'degraded'), (188.0, 're-converged')],
         ext=(3.2, 4.8), vmax=1.5, lab_off=[(8, 3), (8, 6), (8, 4)]),
]


def load_run(run):
    cols = ('timestamp', 'gt_x', 'gt_y', 'gt_yaw', 'ndt_x', 'ndt_y', 'ndt_yaw')
    rows = []
    with open(HERE / f'data/logs/holdout/{run}_a1/long_route_pilot.csv') as fh:
        for r in csv.DictReader(fh):
            if r['phase'] != 'route':
                continue
            try:
                rows.append([float(r[c]) for c in cols])
            except ValueError:
                continue
    a = np.array(rows)
    a = a[np.all(np.isfinite(a), axis=1)]
    d = dict(zip(cols, a.T))
    th = np.radians(d['gt_yaw'])
    ux, uy = np.cos(th), np.sin(th)
    nx, ny = -uy, ux
    cx = d['ndt_x'] - OFF * ux                # frozen route-constant lon offset removed
    cy = d['ndt_y'] - OFF * uy
    dx, dy = cx - d['gt_x'], cy - d['gt_y']
    d['elon'] = dx * ux + dy * uy             # signed, GT frame
    d['elat'] = dx * nx + dy * ny
    d['eyaw'] = (d['ndt_yaw'] - d['gt_yaw'] + 180) % 360 - 180
    d['cx'], d['cy'] = cx, cy
    d['t'] = d['timestamp'] - d['timestamp'][0]
    d['t_abs'] = d['timestamp']
    z = np.load(CACHE / 'holdout_q90.npz', allow_pickle=True)
    m = z['scenario'] == run
    o = np.argsort(z['t'][m])
    ta = z['t'][m][o]
    for ax in ('lon', 'lat', 'yaw'):
        d[f'q_{ax}'] = np.interp(d['t_abs'], ta, z[f'q90_{ax}'][m][o])
    return d


mp = np.load(CACHE / 'town01_crop.npz')
MX, MY, MZ = mp['x'], mp['y'], mp['z']
SUB = np.random.default_rng(0).choice(len(MX), 400_000, replace=False)


def car(ax, x, y, yaw, **kw):
    """Vehicle footprint with a nose, centred at (x,y), heading yaw [rad]."""
    L, W = CAR_L / 2, CAR_W / 2
    pts = np.array([[-L, -W], [L * 0.55, -W], [L, 0], [L * 0.55, W], [-L, W]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    ax.add_patch(Polygon(pts @ R.T + [x, y], closed=True, **kw))


def tile(ax, d, i, ext, axis):
    """Vehicle-centred view at sample i: GT frame, heading up."""
    th = np.radians(d['gt_yaw'][i])
    # world -> GT frame with heading up: rotate by (90deg - yaw)
    rot = np.pi / 2 - th
    c, s = np.cos(rot), np.sin(rot)

    def w2v(x, y):
        x0, y0 = x - d['gt_x'][i], y - d['gt_y'][i]
        return c * x0 - s * y0, s * x0 + c * y0

    hx, hy = ext
    r = np.hypot(hx, hy) + 3
    box = ((MX > d['gt_x'][i] - r) & (MX < d['gt_x'][i] + r)
           & (MY > d['gt_y'][i] - r) & (MY < d['gt_y'][i] + r))
    vx, vy = w2v(MX[box], MY[box])
    g = MZ[box] < 0.5
    ax.scatter(vx[g], vy[g], s=0.15, c='0.90', lw=0, rasterized=True)
    ax.scatter(vx[~g], vy[~g], s=0.15, c='0.70', lw=0, rasterized=True)
    # recent GT path (dashed light) for context
    w = slice(max(0, i - 80), min(len(d['t']), i + 80))
    px, py = w2v(d['gt_x'][w], d['gt_y'][w])
    ax.plot(px, py, color='0.55', lw=0.5, ls=':')
    # NDT-estimated vehicle
    ex, ey = w2v(d['cx'][i], d['cy'][i])
    eyaw = np.radians(d['ndt_yaw'][i]) + rot
    # bounds are centred on the estimate, aligned with the estimated heading
    ql, qt = d['q_lon'][i], d['q_lat'][i]
    for hl, hw, kw in ((ql, qt, dict(fc=C_Q, ec=C_Q, alpha=0.30, lw=0.7)),
                       (BL, BL, dict(fill=False, ec=C_BL, lw=0.9, ls=(0, (3.2, 1.8))))):
        L, W = CAR_L / 2 + hl, CAR_W / 2 + hw
        pts = np.array([[-L, -W], [L, -W], [L, W], [-L, W]])
        cc, ss = np.cos(eyaw), np.sin(eyaw)
        ax.add_patch(Polygon(pts @ np.array([[cc, -ss], [ss, cc]]).T + [ex, ey],
                             closed=True, zorder=3, **kw))
    car(ax, ex, ey, eyaw, fill=False, ec=C_NDT, lw=1.0, zorder=5)
    car(ax, 0, 0, np.pi / 2, fc='0.25', ec='black', lw=0.6, zorder=4)
    ax.set_aspect('equal')
    ax.set_xlim(-hx, hx); ax.set_ylim(-hy, hy)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('0.55')
    # scale bar
    sb = 2 if hx <= 5 else 5
    ax.plot([hx - sb - 0.6, hx - 0.6], [-hy + 0.8, -hy + 0.8], color='k', lw=1.0)
    ax.text(hx - sb / 2 - 0.6, -hy + 1.1, f'{sb} m', ha='center', va='bottom', fontsize=5.8)
    err = abs(d['elat'][i] if axis == 'lat' else d['elon'][i])
    ax.text(0.03, 0.03, f'error {err:.2f} m', transform=ax.transAxes,
            fontsize=6, va='bottom', ha='left')



def make(cs):
    d = load_run(cs['run'])
    axis = cs['axis']
    e = np.abs(d['elat'] if axis == 'lat' else d['elon'])
    t = d['t']
    idx = [int(np.argmin(np.abs(t - ts))) for ts, _ in cs['tiles']]

    fig = plt.figure(figsize=(7.16, 2.55))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.38, 1, 1, 1], wspace=0.06,
                          left=0.005, right=0.995, top=0.93, bottom=0.17)

    # ---- (a) map ----
    axA = fig.add_subplot(gs[0, 0])
    g = MZ[SUB] < 0.5
    axA.scatter(MX[SUB][g], MY[SUB][g], s=0.04, c='0.91', lw=0, rasterized=True)
    axA.scatter(MX[SUB][~g], MY[SUB][~g], s=0.04, c='0.76', lw=0, rasterized=True)
    pts = np.c_[d['gt_x'], d['gt_y']].reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap='YlOrRd', norm=plt.Normalize(0, cs['vmax']), lw=2.4, zorder=3)
    lc.set_array(e[:-1])
    axA.add_collection(lc)
    axA.plot(d['gt_x'][0], d['gt_y'][0], 'o', ms=3.5, color='k', mfc='white', mew=0.8, zorder=4)
    for k, i in enumerate(idx):
        axA.plot(d['gt_x'][i], d['gt_y'][i], 's', ms=4.5, mfc='white', mec='k', mew=0.7, zorder=5)
        axA.annotate(f'b{k+1}', (d['gt_x'][i], d['gt_y'][i]),
                     xytext=cs['lab_off'][k], textcoords='offset points', fontsize=7,
                     arrowprops=dict(arrowstyle='-', lw=0.4, color='0.3', shrinkB=2))
    axA.set_aspect('equal'); axA.set_xlim(62, 408); axA.set_ylim(-225, 28)
    axA.set_xticks([]); axA.set_yticks([])
    for sp in axA.spines.values():
        sp.set_color('0.55')
    axA.plot([80, 130], [-212, -212], color='k', lw=1.1)
    axA.text(105, -206, '50 m', ha='center', va='bottom', fontsize=6.5)
    axA.text(0.02, 0.97, '(a)', transform=axA.transAxes, fontsize=8.5, va='top', weight='bold')
    axA.text(0.98, 0.97, 'start', transform=axA.transAxes, fontsize=6.5, va='top', ha='right')
    cb = fig.colorbar(lc, ax=axA, orientation='horizontal', fraction=0.05, pad=0.025,
                      shrink=0.9, aspect=30)
    cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
    cb.set_label(f'|{"lateral" if axis == "lat" else "longitudinal"} error| [m]',
                 fontsize=6.5, labelpad=1)

    # ---- (b) tiles ----
    for k, (i, (_, lab)) in enumerate(zip(idx, cs['tiles'])):
        ax = fig.add_subplot(gs[0, k + 1])
        tile(ax, d, i, cs['ext'], axis)
        ax.set_title(f'(b{k+1}) {lab}', fontsize=7.5, loc='left', pad=2)

    leg = [Patch(facecolor='0.25', edgecolor='k', label='vehicle (ground truth)'),
           Line2D([], [], color=C_NDT, lw=1.0, label='vehicle as estimated by NDT'),
           Line2D([], [], color=C_BL, lw=0.9, ls=(0, (3.2, 1.8)), label='fixed $3\\sigma$ bound (deployed)'),
           Patch(facecolor=C_Q, alpha=0.35, edgecolor=C_Q, label='MELD-Loc $q^{90}$ bound (ours)')]
    fig.legend(handles=leg, loc='lower center', ncol=4, fontsize=6.8, frameon=False,
               bbox_to_anchor=(0.62, 0.0), handlelength=1.6, columnspacing=1.6)
    fig.savefig(OUT / f"case_{cs['run']}.pdf", dpi=400)
    fig.savefig(OUT / f"case_{cs['run']}.png", dpi=220)
    print('saved', cs['run'])


for cs in CASES:
    make(cs)
