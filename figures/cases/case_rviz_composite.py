#!/usr/bin/env python3
"""Composite: (a) matplotlib map (error-coloured GT track) + (b1-b3) RViz replay tiles."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / 'figures/cases/rviz_replay'))
from replay_node import load_run  # noqa

OUT = HERE / 'figures/out'
CACHE = OUT / '_hero_cache'
from replay_node import ROUTES  # noqa
TILE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
ROUTE = sys.argv[2] if len(sys.argv) > 2 else 'routeA'
RUN, AXIS = 'ho_ls08_r52', 'lat'
TILES = {'routeA': [(50.0, 'b1', 'normal'), (98.0, 'b2', 'degraded'), (188.0, 'b3', 're-converged')],
         'routeB': [(40.0, 'b1', 'normal'), (102.0, 'b2', 'degraded, covered'),
                    (124.0, 'b3', 'degraded, missed')]}[ROUTE]
# map extents (x0, x1, y0, y1) and 50 m scale-bar anchor (x, y)
VIEW = {'routeA': ((62, 408, -225, 28), (80, -212)),
        'routeB': ((-25, 212, -325, -172), (-10, -318))}[ROUTE]
SCALE = 75.0                         # px per metre in the RViz captures
CROP = (95, 127, 903, 935)           # render area inside the 1000x1000 window (square)
C_NDT, C_BL, C_Q = '#c8102e', '#1f5fbf', '#e87d0d'
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 7})

d = load_run(RUN, ROUTE)
th = np.radians(d['gt_yaw']); ux, uy = np.cos(th), np.sin(th)
dx, dy = d['cx'] - d['gt_x'], d['cy'] - d['gt_y']
e = np.abs(-dx*uy + dy*ux) if AXIS == 'lat' else np.abs(dx*ux + dy*uy)
t = d['t']
idx = [int(np.argmin(np.abs(t - ts))) for ts, _, _ in TILES]
mp = np.load(CACHE / ROUTES[ROUTE]['map']); MX, MY, MZ = mp['x'], mp['y'], mp['z']
SUB = np.random.default_rng(0).choice(len(MX), 400_000, replace=False)

fig = plt.figure(figsize=(7.16, 2.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1.38, 1, 1, 1], wspace=0.05,
                      left=0.005, right=0.995, top=0.93, bottom=0.17)
axA = fig.add_subplot(gs[0, 0])
g = MZ[SUB] < 0.5
axA.scatter(MX[SUB][g], MY[SUB][g], s=0.04, c='0.91', lw=0, rasterized=True)
axA.scatter(MX[SUB][~g], MY[SUB][~g], s=0.04, c='0.76', lw=0, rasterized=True)
pts = np.c_[d['gt_x'], d['gt_y']].reshape(-1, 1, 2)
seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(seg, cmap='YlOrRd', norm=plt.Normalize(0, 1.5), lw=2.4, zorder=3)
lc.set_array(e[:-1]); axA.add_collection(lc)
axA.plot(d['gt_x'][0], d['gt_y'][0], 'o', ms=3.5, color='k', mfc='white', mew=0.8, zorder=4)
for (i, (_, tag, _), off) in zip(idx, TILES, [(8, 3), (8, 8), (-16, -10)]):
    axA.plot(d['gt_x'][i], d['gt_y'][i], 's', ms=4.5, mfc='white', mec='k', mew=0.7, zorder=5)
    axA.annotate(tag, (d['gt_x'][i], d['gt_y'][i]), xytext=off, textcoords='offset points',
                 fontsize=7, arrowprops=dict(arrowstyle='-', lw=0.4, color='0.3', shrinkB=2))
(x0, x1, y0, y1), (sx, sy) = VIEW
axA.set_aspect('equal'); axA.set_xlim(x0, x1); axA.set_ylim(y0, y1)
axA.set_xticks([]); axA.set_yticks([])
for sp in axA.spines.values(): sp.set_color('0.55')
axA.plot([sx, sx + 50], [sy, sy], color='k', lw=1.1); axA.text(sx + 25, sy + 5, '50 m', ha='center', va='bottom', fontsize=6.5)
axA.text(0.02, 0.97, '(a)', transform=axA.transAxes, fontsize=8.5, va='top', weight='bold')
st_off, st_ha = ((-4, 5), 'right') if ROUTE == 'routeA' else ((5, 4), 'left')
axA.annotate('start', (d['gt_x'][0], d['gt_y'][0]), xytext=st_off, textcoords='offset points', fontsize=6.5, ha=st_ha)
cb = fig.colorbar(lc, ax=axA, orientation='horizontal', fraction=0.05, pad=0.025, shrink=0.9, aspect=30)
cb.ax.tick_params(labelsize=6, length=1.5, pad=1); cb.set_label('|lateral error| [m]', fontsize=6.5, labelpad=1)

for k, (i, (_, tag, lab)) in enumerate(zip(idx, TILES)):
    ax = fig.add_subplot(gs[0, k + 1])
    im = Image.open(TILE_DIR / f'rv_{ROUTE}_{tag}.png').crop(CROP)
    ax.imshow(im, interpolation='lanczos')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('0.55')
    W, H = im.size
    ax.plot([W - 5*SCALE - 30, W - 30], [H - 40, H - 40], color='k', lw=1.2)
    ax.text(W - 2.5*SCALE - 30, H - 50, '5 m', ha='center', va='bottom', fontsize=6.5)
    ax.text(0.03, 0.03, f'error {e[i]:.2f} m', transform=ax.transAxes, fontsize=6.5, va='bottom')
    ax.set_title(f'({tag}) {lab}', fontsize=7.5, loc='left', pad=2)

leg = [Patch(facecolor='0.25', edgecolor='k', label='vehicle (ground truth)'),
       Patch(facecolor=C_NDT, alpha=0.6, edgecolor=C_NDT, label='vehicle as estimated by NDT'),
       Line2D([], [], color=C_BL, lw=1.2, label='fixed-covariance $3\\sigma$ reference'),
       Line2D([], [], color=C_Q, lw=1.6, label='MELD-Loc $90\\%$ bound')]
fig.legend(handles=leg, loc='lower center', ncol=4, fontsize=6.8, frameon=False,
           bbox_to_anchor=(0.62, 0.0), handlelength=1.6, columnspacing=1.6)
fig.savefig(OUT / f'case_rviz_{ROUTE}_{RUN}.pdf', dpi=300)
fig.savefig(OUT / f'case_rviz_{ROUTE}_{RUN}.png', dpi=220)
print('saved')
