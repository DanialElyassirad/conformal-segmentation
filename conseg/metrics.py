"""Segmentation metrics and risk-coverage analysis."""
import numpy as np
from scipy import ndimage

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ----------------------------------------------------------------------------- HD95
def _surface(m):
    return m & ~ndimage.binary_erosion(m, border_value=0)


def hd95(a, b, spacing, empty="nan"):
    """95th percentile symmetric Hausdorff distance between two binary masks (in spacing units)."""
    a = a.astype(bool); b = b.astype(bool)
    if not a.any() or not b.any():
        if empty == "diagonal":
            return float(np.sqrt(sum((n * s) ** 2 for n, s in zip(a.shape, spacing))))
        return float("nan")
    union = a | b
    lo = [int(x.min()) for x in np.nonzero(union)]
    hi = [int(x.max()) + 1 for x in np.nonzero(union)]
    sl = tuple(slice(max(l - 1, 0), h + 1) for l, h in zip(lo, hi))
    a, b = a[sl], b[sl]
    sa, sb = _surface(a), _surface(b)
    da = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


# ----------------------------------------------------------------------------- risk-coverage
def rc_curve(score, risk, n_points=100):
    """Cases retained in ascending uncertainty order. Returns coverage grid and mean risk of retained cases."""
    score = np.asarray(score, float); risk = np.asarray(risk, float)
    keep = ~(np.isnan(score) | np.isnan(risk))
    score, risk = score[keep], risk[keep]
    n = len(score)
    order = np.argsort(score, kind="mergesort")
    csum = np.cumsum(risk[order])
    ks = np.unique(np.clip(np.round(np.linspace(1, n, min(n_points, n))).astype(int), 1, n))
    return ks / n, csum[ks - 1] / ks


def aurcc(cov, mean_risk):
    """Area under the risk-coverage curve, normalized by the coverage range (lower is better)."""
    if len(cov) < 2:
        return float(mean_risk[-1])
    return float(_trapz(mean_risk, cov) / (cov[-1] - cov[0]))


def aurcc_of(score, risk):
    return aurcc(*rc_curve(score, risk))


def retention_means(score, risk, levels):
    score = np.asarray(score, float); risk = np.asarray(risk, float)
    keep = ~(np.isnan(score) | np.isnan(risk))
    s, r = score[keep], risk[keep]
    order = np.argsort(s, kind="mergesort")
    out = {}
    for lv in levels:
        k = max(1, int(round(lv * len(s))))
        out[lv] = float(r[order][:k].mean())
    return out


def risk_coverage_analysis(score, risk, n_boot=1000, n_perm=10000, seed=42):
    """AURCC of uncertainty ranking vs random ranking, bootstrap CIs and a two-sided permutation P value."""
    score = np.asarray(score, float); risk = np.asarray(risk, float)
    keep = ~(np.isnan(score) | np.isnan(risk))
    s, r = score[keep], risk[keep]
    n = len(s)
    if n < 3:
        return {"n": n}
    rng = np.random.default_rng(seed)
    a_obs = aurcc_of(s, r)
    a_rand = float(r.mean())                       # exact expectation under random ranking
    b_a, b_d = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, n, n)
        b_a[i] = aurcc_of(s[j], r[j])
        b_d[i] = b_a[i] - r[j].mean()
    perm = np.empty(n_perm)
    for i in range(n_perm):
        perm[i] = aurcc_of(rng.permutation(n).astype(float), r)
    centre = perm.mean()
    p_two = (1 + np.sum(np.abs(perm - centre) >= abs(a_obs - centre))) / (n_perm + 1)
    p_one = (1 + np.sum(perm <= a_obs)) / (n_perm + 1)
    return {
        "n": n, "aurcc": a_obs, "aurcc_ci": tuple(np.percentile(b_a, [2.5, 97.5])),
        "aurcc_random": a_rand, "delta": a_obs - a_rand, "delta_ci": tuple(np.percentile(b_d, [2.5, 97.5])),
        "p_two_sided": float(p_two), "p_one_sided": float(p_one),
    }


def format_p(p):
    if p is None or np.isnan(p):
        return "n/a"
    return "P < .001" if p < 0.001 else f"P = {p:.3f}".replace("0.", ".")
