#!/usr/bin/env python3
"""MELD-Loc core library (registered design).

Structure: score-free core F (monotone in t) with cluster-wise OOF logits -> one family
      re-weighting -> weighted F -> residual adapter G on the matcher score (init_score = OOF)
      -> lambda chosen once on dev-val -> isotonic calibrator h -> conformal-style 90%
      one-sided upper shift (recipe-group residual quantile) -> frozen serialization.

Properties:
  - the random component of the threshold samples is a deterministic sequence keyed by the anchor id
  - lambda, h and the conformal shift are fitted on the development-validation split
  - every fitted asset is serialized (F, G, lambda, h, shift, t_max, grids, feature ids, family weights, version hash)

The resampling unit of the calibrator and the conformal shift is the recipe group
(cluster_id where an index predates recipe groups).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# ---- frozen constants ----
T1 = {"lon": 2.5, "lat": 0.85, "yaw": 2.4337}
T2 = {"lon": 5.0, "lat": 1.75, "yaw": 4.858}
W3 = {"lon": 8.1739, "lat": 1.6117, "yaw": 9.6618}   # 3W (upper edge of the mid band = divergence threshold)
K = 10                                                # threshold samples per anchor (4/2/2/2)
LAM_GRID = (0.0, 0.25, 0.5, 1.0)
WCAP = (0.5, 2.0)                                     # family-weight cap
TMAX_Q = 0.995                                        # t_max = train-only 99.5th percentile
N_TGRID = 41                                          # curve evaluation grid (asinh-uniform)
N_RGRID = 65                                          # integration grid of R: uniform on [T1, 3W], end points included
CONTROL = {"translate_y", "yaw_rotate", "translate_x"}
FEATURE_SET_ID = "A6-0"                               # {2 s, 8 s} x 5 signals = 120-d

GBDT_DEFAULT = dict(n_estimators=400, num_leaves=63, learning_rate=0.05,
                    min_child_samples=50, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, reg_lambda=1.0, random_state=20260803,
                    verbose=-1, n_jobs=12)

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64: vectorized deterministic hash (source of the anchor-keyed sequences)."""
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & _M64
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
    return z ^ (z >> np.uint64(31))


def anchor_keys(scenario: np.ndarray, t_anchor: np.ndarray) -> np.ndarray:
    """Per-anchor 64-bit key: hash(scenario_id) xor time in ms; independent of order and run."""
    sh = {s: int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
          for s in np.unique(scenario)}
    a = np.array([sh[s] for s in scenario], dtype=np.uint64)
    b = np.round(t_anchor * 1000.0).astype(np.int64).astype(np.uint64)
    return _mix64(a ^ _mix64(b))


def _u01(keys: np.ndarray, comp: int) -> np.ndarray:
    return (_mix64(keys ^ np.uint64(0xA5A5A5A5 + comp)) >> np.uint64(11)).astype(
        np.float64) / float(1 << 53)


def make_pairs(X, ycap, t_max, t1, t2, w3, keys):
    """Threshold augmentation (x,t), K=10 per anchor: 4 asinh-uniform / 2 mid-band / 1 functional threshold + 1 tail / 2 boundary.

    All random components are anchor-key deterministic; the K pairs of one anchor share weight 1/K.
    Returns X_pairs (features + one threshold column), labels 1{Ycap>t}, weights, anchor indices.
    """
    n = len(X)
    at = float(np.arcsinh(t_max))
    Ts = np.empty((n, K))
    for c in range(4):
        Ts[:, c] = np.sinh(_u01(keys, c) * at)
    Ts[:, 4] = t1 + _u01(keys, 4) * (w3 - t1)
    Ts[:, 5] = t1 + _u01(keys, 5) * (w3 - t1)
    Ts[:, 6] = np.array([t1, t2, w3])[(keys % np.uint64(3)).astype(int)]
    Ts[:, 7] = w3 + _u01(keys, 7) * max(t_max - w3, 1e-6)
    Ts[:, 8] = 0.0
    Ts[:, 9] = t_max
    Xp = np.repeat(X, K, axis=0)
    tf = (np.arcsinh(Ts.reshape(-1)) / max(at, 1e-9))[:, None].astype(np.float32)
    lab = (np.repeat(ycap, K) > Ts.reshape(-1)).astype(np.int32)
    w = np.full(len(lab), 1.0 / K)
    aidx = np.repeat(np.arange(n), K)
    return np.hstack([Xp, tf]).astype(np.float32), lab, w, aidx


def cluster_folds(clusters: np.ndarray, n_folds: int = 5, seed: int = 20260803):
    uc = np.unique(clusters)
    order = np.random.default_rng(seed).permutation(len(uc))
    fold_of = {uc[i]: r % n_folds for r, i in enumerate(order)}
    return np.array([fold_of[c] for c in clusters]), fold_of


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def family_weights(oof_p, lab, fam_pairs, cap=WCAP):
    """One family re-weighting from the unweighted OOF log-loss: w_f ~ (L_f/median)^0.5, capped to [0.5, 2]."""
    eps = 1e-7
    ll = -(lab * np.log(oof_p + eps) + (1 - lab) * np.log(1 - oof_p + eps))
    fams = sorted(set(fam_pairs))
    Lf = {f: float(ll[fam_pairs == f].mean()) for f in fams}
    med = float(np.median(list(Lf.values())))
    return {f: float(np.clip((Lf[f] / med) ** 0.5, *cap)) for f in fams}


def group_equal_weights(groups: np.ndarray) -> np.ndarray:
    """Weights giving every recipe group equal total weight (the resampling unit of calibration and conformal shift)."""
    _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w / w.sum() * len(w)


def first_below(S, tg, thr):
    """First t at which S(t) <= thr (linear interpolation): the quantile inverse of the curve."""
    out = np.empty(len(S))
    for i in range(len(S)):
        row = S[i]
        if row[0] <= thr:
            out[i] = tg[0]
            continue
        m = row <= thr
        if not m.any():
            out[i] = tg[-1]
            continue
        k = int(np.argmax(m))
        s0, s1 = row[k - 1], row[k]
        f = 0.0 if s0 == s1 else (s0 - thr) / (s0 - s1)
        out[i] = tg[k - 1] + f * (tg[k] - tg[k - 1])
    return out


def prob_at(S, tg, t):
    """S(x, t) by linear interpolation on the grid."""
    j = np.searchsorted(tg, t)
    if j <= 0:
        return S[:, 0]
    if j >= len(tg):
        return S[:, -1]
    f = (t - tg[j - 1]) / (tg[j] - tg[j - 1])
    return S[:, j - 1] * (1 - f) + S[:, j] * f


class MeldLoc:
    """Frozen model of one axis: holds every fitted asset and evaluates z = F + lambda * G."""

    def __init__(self, axis, F, G, lam, iso, conf_shift, t_max, t_grid,
                 fam_w, xa_cols, xs_cols, meta):
        self.axis = axis
        self.F, self.G, self.lam, self.iso = F, G, lam, iso
        self.conf_shift = conf_shift          # additive shift in asinh space (90% upper bound)
        self.t_max, self.t_grid = t_max, t_grid
        self.fam_w = fam_w
        self.xa_cols, self.xs_cols = xa_cols, xs_cols
        self.meta = meta

    # ---- curve ----
    def _eval_S(self, X, ts, lam, calibrated):
        """Raw curve values on an ascending t sequence, before the monotone projection."""
        at = float(np.arcsinh(self.t_max))
        Xa = X[:, self.xa_cols].astype(np.float32)
        Xs = X[:, self.xs_cols].astype(np.float32)
        S = np.empty((len(X), len(ts)), np.float32)
        for j, tv in enumerate(ts):
            tf = np.full((len(X), 1), np.arcsinh(tv) / at, np.float32)
            z = self.F.predict(np.hstack([Xa, tf]), raw_score=True)
            if lam != 0.0:
                z = z + lam * self.G.predict(np.hstack([Xs, tf]), raw_score=True)
            p = sigmoid(z)
            S[:, j] = self.iso.predict(p) if (calibrated and self.iso is not None) else p
        return S

    def curve(self, X, lam=None, calibrated=True, return_violation=False):
        lam = self.lam if lam is None else lam
        S = self._eval_S(X, self.t_grid, lam, calibrated)
        viol = float((np.diff(S, axis=1) > 1e-6).mean())
        S = np.minimum.accumulate(S, axis=1)      # numerical safeguard (h is monotone, so this is a no-op in principle)
        return (S, viol) if return_violation else S

    # ---- the four decision outputs ----
    def outputs(self, X, lam=None):
        lam = self.lam if lam is None else lam
        S, viol = self.curve(X, lam=lam, return_violation=True)
        tg = self.t_grid
        t1, t2, w3 = T1[self.axis], T2[self.axis], W3[self.axis]
        e_med = first_below(S, tg, 0.5)
        q90_raw = first_below(S, tg, 0.10)
        q90_conf = np.sinh(np.arcsinh(q90_raw) + self.conf_shift)
        # R = integral of S over [T1, 3W] on its own uniform grid (end points included, N_RGRID fixed)
        r_grid = np.linspace(t1, w3, N_RGRID)
        S_R = np.minimum.accumulate(self._eval_S(X, r_grid, lam, True), axis=1)
        R = np.trapezoid(S_R, r_grid, axis=1)
        return {"S": S, "viol": viol, "e_med": e_med, "q90_raw": q90_raw,
                "q90_conf": q90_conf,
                "p_T1": prob_at(S, tg, t1), "p_T2": prob_at(S, tg, t2),
                "p_3W": prob_at(S, tg, w3), "R": R}

    def R_quantile_band(self, X, q_lo, q_hi):
        """Companion for the longitudinal axis: integration over the quantile interval [q85, q99]."""
        S = self.curve(X)
        m = (self.t_grid >= q_lo) & (self.t_grid <= q_hi)
        return np.trapezoid(S[:, m], self.t_grid[m], axis=1)

    # ---- serialization ----
    def save(self, path: Path):
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"axis": self.axis, "F": self.F, "G": self.G, "lam": self.lam,
                     "iso": self.iso, "conf_shift": self.conf_shift,
                     "t_max": self.t_max, "t_grid": self.t_grid,
                     "fam_w": self.fam_w, "xa_cols": self.xa_cols,
                     "xs_cols": self.xs_cols, "meta": self.meta}, path)

    @staticmethod
    def load(path: Path) -> "MeldLoc":
        import joblib
        d = joblib.load(path)
        return MeldLoc(d["axis"], d["F"], d["G"], d["lam"], d["iso"], d["conf_shift"],
                       d["t_max"], d["t_grid"], d["fam_w"], d["xa_cols"], d["xs_cols"],
                       d["meta"])


def version_hash() -> str:
    src = Path(__file__).read_bytes()
    cfg = json.dumps({"K": K, "LAM_GRID": LAM_GRID, "WCAP": WCAP, "TMAX_Q": TMAX_Q,
                      "N_TGRID": N_TGRID, "N_RGRID": N_RGRID, "GBDT": GBDT_DEFAULT,
                      "FEATURE_SET_ID": FEATURE_SET_ID}, sort_keys=True).encode()
    return hashlib.sha256(src + cfg).hexdigest()[:16]


# ---- shared metrics ----
def spearman(a, b):
    if len(a) < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def auroc(s, l):
    p, n = s[l == 1], s[l == 0]
    if len(p) < 2 or len(n) < 2:
        return float("nan")
    a = np.concatenate([p, n])
    o = np.argsort(a, kind="mergesort")
    sa = a[o]
    r = np.empty(len(a))
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def paired_group_ci(metric_fn, a, b, groups, n_boot=600, seed=20260803):
    """paired group(cluster/recipe_group) bootstrap 95% CI of metric(a)−metric(b)."""
    uc = np.unique(groups)
    idx = {c: np.where(groups == c)[0] for c in uc}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        bs = rng.choice(uc, len(uc), replace=True)
        ii = np.concatenate([idx[c] for c in bs])
        va, vb = metric_fn(a, ii), metric_fn(b, ii)
        if np.isfinite(va) and np.isfinite(vb):
            out.append(va - vb)
    if len(out) < 50:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))
