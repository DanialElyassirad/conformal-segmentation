"""Calibration: Dice-optimal threshold, normalization and conformal threshold (NCST) per target."""
import datetime
import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import __version__, core, io
from .data import calibration_pool, load_case

log = logging.getLogger("conseg")


def _map(fn, jobs, workers):
    if workers and workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, jobs))
    return [fn(j) for j in jobs]


def _imap(fn, jobs, workers):
    """Like _map, but yields results one at a time, so large per-case results (pass 2 histograms are
    8 MB per case and target) are never all held in memory."""
    if workers and workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i in range(0, len(jobs), 2 * workers):
                yield from ex.map(fn, jobs[i:i + 2 * workers])
    else:
        for j in jobs:
            yield fn(j)


def resolve_targets(cfg, label_values, has_ch, n_channels):
    """Returns ({name: label values}, {name: probability channels})."""
    targets = cfg["labels"]
    if not targets:
        vals = sorted(v for v in label_values if v > 0)
        if not vals:
            raise ValueError("The calibration labels contain no foreground voxels")
        targets = {f"label_{v}": [v] for v in vals}
        log.info("No labels configured; using one target per label value: %s", targets)
    chans = {k: list((cfg.get("channels") or {}).get(k, v)) for k, v in targets.items()}
    if not has_ch:
        if len(targets) != 1:
            raise ValueError("Single-channel probability maps describe one target; configure exactly one entry in 'labels'")
    else:
        for name, ch in chans.items():
            bad = [m for m in ch if m >= n_channels]
            if bad:
                raise ValueError(f"Target '{name}' uses channels {bad}, but the probability maps have only "
                                 f"{n_channels} channels (0-{n_channels - 1}). For region-based or sigmoid models "
                                 f"whose channels are not the label values, set 'channels' in the config.")
    return {k: list(v) for k, v in targets.items()}, chans


# ----------------------------------------------------------------------------- per-case workers
def _probe(job):
    cid, pp, lp, cfg = job
    c = load_case(cid, pp, lp, cfg)
    return {"id": cid, "values": np.unique(c["label"]).tolist(), "has_ch": c["has_ch"],
            "n_channels": int(c["probs"].shape[0]) if c["has_ch"] else 1, "kind": c["kind"],
            "raw_kind": c["raw_kind"], "transposed": c["transposed"], "ndim": int(c["label"].ndim),
            "units": c["units"]}


def _pass_threshold(job):
    cid, pp, lp, cfg, targets, chans = job
    c = load_case(cid, pp, lp, cfg)
    out = {}
    for name, mem in targets.items():
        p = core.target_probability(c["probs"], c["has_ch"], chans[name], c["kind"])
        g = core.target_mask(c["label"], mem)
        out[name] = {"dice": core.dice_per_threshold(p, g), "fg": int(g.sum()), "n": int(g.size)}
    return out


def _pass_scores(job):
    cid, pp, lp, cfg, targets, chans, thr, nbins = job
    c = load_case(cid, pp, lp, cfg)
    out = {}
    for name, mem in targets.items():
        p = core.target_probability(c["probs"], c["has_ch"], chans[name], c["kind"])
        g = core.target_mask(c["label"], mem)
        sq = core.StreamingQuantile(nbins)
        sq.add(core.nonconformity(core.normalize(p, thr[name]), g))
        out[name] = sq.hist
    return out


def _pass_stats(job):
    cid, pp, lp, cfg, targets, chans, thr, ncst = job
    c = load_case(cid, pp, lp, cfg)
    out = {}
    for name, mem in targets.items():
        p = core.target_probability(c["probs"], c["has_ch"], chans[name], c["kind"])
        g = core.target_mask(c["label"], mem)
        st, _ = core.case_statistics(p, thr[name], ncst[name], g)
        out[name] = st
    return out


# ----------------------------------------------------------------------------- alpha selection
def _choose_alpha(sq, fg_fraction, ratio, requested):
    """Fixed alpha, or automatic: start at ratio x foreground fraction and adjust until the NCST is non-degenerate."""
    if requested != "auto":
        a = float(requested)
        ncst = sq.conformal_threshold(a)
        if ncst <= 0.01 or ncst >= 0.99:
            log.warning("alpha=%g gives a degenerate NCST (%.4f); consider another alpha", a, ncst)
        return a, ncst, "fixed"
    a = core.round_sig(max(min(ratio * fg_fraction, 0.2), 1e-7))
    ncst = sq.conformal_threshold(a)
    steps = 0
    while ncst >= 0.99 and a < min(0.5 * fg_fraction, 0.2) and steps < 30:
        a = core.round_sig(a * 2); ncst = sq.conformal_threshold(a); steps += 1
    while ncst <= 0.01 and a > 1e-7 and steps < 60:
        a = core.round_sig(a / 2); ncst = sq.conformal_threshold(a); steps += 1
    if ncst <= 0.01 or ncst >= 0.99:
        log.warning("Automatic alpha could not find a non-degenerate NCST (alpha=%g, NCST=%.4f)", a, ncst)
    return a, ncst, "auto"


# ----------------------------------------------------------------------------- main entry point
def calibrate(cfg, pairs=None):
    """Calibrate on labeled cases. Returns the calibration dictionary (also saved as JSON by the caller)."""
    test_pairs = {}
    if pairs is None:
        pairs, test_pairs = calibration_pool(cfg)
    ids = sorted(pairs)
    W = cfg.get("workers", 1)
    log.info("Calibrating on %d cases", len(ids))

    probes = _map(_probe, [(c, *pairs[c], cfg) for c in ids], W)
    first = probes[0]
    for pr in probes[1:]:
        if pr["has_ch"] != first["has_ch"] or pr["n_channels"] != first["n_channels"]:
            raise ValueError(f"Case {pr['id']} has a different probability layout from {first['id']}")
    values = sorted(set(v for pr in probes for v in pr["values"]))
    targets, chans = resolve_targets(cfg, values, first["has_ch"], first["n_channels"])

    # pass 1: Dice-optimal threshold
    res = _map(_pass_threshold, [(c, *pairs[c], cfg, targets, chans) for c in ids], W)
    thr, info = {}, {}
    for name in targets:
        dices = np.stack([r[name]["dice"] for r in res])
        mean_d = dices.mean(0)
        fg = sum(r[name]["fg"] for r in res); n = sum(r[name]["n"] for r in res)
        if cfg["calibration"]["threshold"] == "auto":
            k = int(np.argmax(mean_d)); t = float(core.THRESHOLDS[k]); mode = "auto"
        else:
            t = float(cfg["calibration"]["threshold"]); mode = "fixed"
            k = int(np.argmin(np.abs(core.THRESHOLDS - t)))
        thr[name] = t
        info[name] = {"members": targets[name], "channels": chans[name], "threshold": t, "threshold_mode": mode,
                      "calibration_mean_dice": float(mean_d[k]), "fg_fraction": fg / n, "n_voxels": n,
                      "n_cases": len(ids)}
        if fg == 0:
            raise ValueError(f"Target '{name}' has no foreground voxels in the calibration labels")
        log.info("[%s] threshold %.2f (%s), calibration mean Dice %.4f, foreground fraction %.6f",
                 name, t, mode, mean_d[k], fg / n)

    # pass 2: pooled nonconformity scores -> NCST (histograms summed as they arrive)
    nb = cfg["calibration"]["nbins"]
    sqs = {name: core.StreamingQuantile(nb) for name in targets}
    for r in _imap(_pass_scores, [(c, *pairs[c], cfg, targets, chans, thr, nb) for c in ids], W):
        for name in targets:
            sqs[name].hist += r[name]
    ncst = {}
    alpha_cfg = cfg["calibration"]["alpha"]
    for name in targets:
        sq = sqs[name]
        req = alpha_cfg.get(name, "auto") if isinstance(alpha_cfg, dict) else alpha_cfg
        a, q, mode = _choose_alpha(sq, info[name]["fg_fraction"], cfg["calibration"]["alpha_ratio"], req)
        ncst[name] = q
        info[name].update({"alpha": a, "alpha_mode": mode, "ncst": q, "target_coverage": 1 - a})
        log.info("[%s] alpha %g (%s) -> NCST %.6f", name, a, mode, q)

    # pass 3: calibration coverage and UR distribution
    res = _map(_pass_stats, [(c, *pairs[c], cfg, targets, chans, thr, ncst) for c in ids], W)
    for name in targets:
        urs = np.array([r[name]["UR"] for r in res], float)
        cov_num = sum(r[name]["coverage"] * r[name]["n_voxels"] for r in res)
        cov_den = sum(r[name]["n_voxels"] for r in res)
        finite = urs[~np.isnan(urs)]
        info[name]["calibration_coverage"] = cov_num / cov_den
        info[name]["ur_undefined_fraction"] = float(np.isnan(urs).mean())
        info[name]["ur_percentiles"] = ({str(p): float(np.percentile(finite, p)) for p in range(1, 100)}
                                        if len(finite) else {})

    return {
        "conseg_version": __version__,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "input": {"kind": first["kind"], "raw_kind": first["raw_kind"], "has_channels": first["has_ch"],
                  "n_channels": first["n_channels"], "spatial_ndim": first["ndim"],
                  "transposed": first["transposed"], "units": first["units"]},
        "targets": info,
        "calibration_cases": ids,
        "test_cases": sorted(test_pairs),
        "_test_pairs": test_pairs,
    }


def save_calibration(calib, path):
    out = {k: v for k, v in calib.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def load_calibration(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expect_from(calib):
    i = calib["input"]
    return {"spatial_ndim": i["spatial_ndim"], "has_channels": i["has_channels"], "n_channels": i["n_channels"],
            "transposed": i["transposed"], "raw_kind": i["raw_kind"]}
