"""Core conformal segmentation method (threshold, normalization, split conformal prediction, uncertainty ratio)."""
import math

import numpy as np

THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)   # candidate BMOT values
EPS = 1e-6


# ----------------------------------------------------------------------------- targets
def target_probability(probs, has_ch, members, kind):
    """Foreground probability of a (possibly composite) target.
    softmax channels: sum of member channels; independent sigmoid channels: maximum of members."""
    if not has_ch:
        return probs
    sel = probs[list(members)]
    p = sel.sum(0) if kind == "softmax" else sel.max(0)
    return np.clip(p, 0.0, 1.0)


def target_mask(label, members):
    return np.isin(label, list(members))


# ----------------------------------------------------------------------------- Dice-optimal threshold
def dice_per_threshold(p, g, thresholds=THRESHOLDS):
    """Exact Dice of (p > t) vs g for every threshold t, from one pass over the voxels."""
    idx = np.searchsorted(thresholds, p.ravel(), side="left")        # number of thresholds < p
    g = g.ravel().astype(bool)
    n = len(thresholds) + 1
    c1 = np.bincount(idx[g], minlength=n).astype(np.float64)
    c0 = np.bincount(idx[~g], minlength=n).astype(np.float64)
    # voxel predicted positive at threshold k  <=>  idx > k
    tp = np.cumsum(c1[::-1])[::-1][1:]
    fp = np.cumsum(c0[::-1])[::-1][1:]
    fn = c1.sum() - tp
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(den > 0, 2 * tp / den, 1.0)
    return d


def dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * np.logical_and(a, b).sum() / s)


# ----------------------------------------------------------------------------- normalization
def normalize(p, t):
    """Piecewise-linear map so that the threshold t maps to 0.5 and the case min/max map to 0/1."""
    t = min(max(float(t), EPS), 1 - EPS)
    ymin, ymax = float(p.min()), float(p.max())
    ymax = max(ymax, t + EPS)
    ymin = min(ymin, t - EPS)
    out = np.empty_like(p, dtype=np.float32)
    hi = p > t
    out[hi] = (p[hi] - t) / (2 * (ymax - t)) + 0.5
    out[~hi] = (p[~hi] - t) / (2 * (t - ymin)) + 0.5
    return np.clip(out, 0.0, 1.0)


def nonconformity(q, g):
    """1 minus the normalized score of the true class."""
    return np.where(g, 1.0 - q, q).astype(np.float32)


# ----------------------------------------------------------------------------- streaming quantile
class StreamingQuantile:
    """Histogram-based quantile of scores in [0, 1] pooled over many voxels (memory independent of n)."""

    def __init__(self, nbins=1_000_000):
        self.nbins = int(nbins)
        self.hist = np.zeros(self.nbins, dtype=np.int64)

    def add(self, scores):
        i = np.minimum((np.asarray(scores, np.float64).ravel() * self.nbins).astype(np.int64), self.nbins - 1)
        self.hist += np.bincount(i, minlength=self.nbins)

    def merge(self, other):
        self.hist += other.hist

    @property
    def n(self):
        return int(self.hist.sum())

    def conformal_threshold(self, alpha):
        """Upper edge of the bin holding the ceil((n+1)(1-alpha))-th smallest score (conservative)."""
        n = self.n
        k = math.ceil((n + 1) * (1 - alpha))
        if k > n:
            return 1.0
        b = int(np.searchsorted(np.cumsum(self.hist), k))
        return min((b + 1) / self.nbins, 1.0)


# ----------------------------------------------------------------------------- prediction sets
def conformal_map(q, ncst):
    """0 = certain background, 1 = uncertain (both classes or neither), 2 = certain foreground."""
    in1 = (1.0 - q) <= ncst
    in0 = q <= ncst
    m = np.ones(q.shape, np.uint8)
    m[in1 & ~in0] = 2
    m[in0 & ~in1] = 0
    return m, in1, in0


def case_statistics(p, t, ncst, g=None):
    q = normalize(p, t)
    cmap, in1, in0 = conformal_map(q, ncst)
    n_unc = int((cmap == 1).sum()); n_c1 = int((cmap == 2).sum()); n_c0 = int((cmap == 0).sum())
    st = {
        "n_uncertain": n_unc, "n_certain_fg": n_c1, "n_certain_bg": n_c0,
        "UR": (n_unc / n_c1) if n_c1 else float("nan"),
        "pred_fg_voxels": int((p > t).sum()),
    }
    if g is not None:
        gb = g.astype(bool)
        st["coverage"] = float(np.where(gb, in1, in0).mean())
        st["n_voxels"] = int(gb.size)
        st["ref_fg_voxels"] = int(gb.sum())
    return st, cmap


def round_sig(x, sig=1):
    if x <= 0:
        return x
    return float(f"{x:.{sig}g}")
