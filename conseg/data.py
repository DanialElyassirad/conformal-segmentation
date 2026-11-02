"""Case discovery, pairing, splitting and loading."""
import csv
import logging

import numpy as np

from . import io
from .config import resolve_paths

log = logging.getLogger("conseg")


def _index(cfg, section):
    sec = cfg[section]
    inp = cfg["input"]
    probs = io.index_probabilities(resolve_paths(cfg, sec["probabilities"]), inp["prob_suffix"])
    labels = None
    if sec.get("labels"):
        labels = io.index_images(resolve_paths(cfg, sec["labels"]), inp["label_suffix"])
    images = None
    if sec.get("images"):
        images = io.index_images(resolve_paths(cfg, sec["images"]), None, strip_channel=True)
    return probs, labels, images


def calibration_pool(cfg):
    """Labeled cases for calibration; with calibration.split, also returns the held-out labeled test cases."""
    if not cfg["calibration"]["probabilities"] or not cfg["calibration"]["labels"]:
        raise ValueError("calibration.probabilities and calibration.labels are required")
    probs, labels, _ = _index(cfg, "calibration")
    pairs = io.pair_cases(probs, labels, what="calibration cases")
    split = cfg["calibration"]["split"]
    if not split:
        return pairs, {}
    ids = sorted(pairs)
    groups = {c: c for c in ids}
    pm = cfg["calibration"].get("patient_map")
    if pm:
        with open(resolve_paths(cfg, pm)[0], newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("case") in groups:
                    groups[row["case"]] = row.get("patient") or row["case"]
    units = sorted(set(groups.values()))
    rng = np.random.default_rng(cfg["calibration"]["seed"])
    rng.shuffle(units)
    n_cal = max(1, int(round(float(split) * len(units))))
    cal_units = set(units[:n_cal])
    cal = {c: pairs[c] for c in ids if groups[c] in cal_units}
    test = {c: pairs[c] for c in ids if groups[c] not in cal_units}
    log.info("Split %d labeled cases (%d units) into %d calibration and %d test cases",
             len(ids), len(units), len(cal), len(test))
    return cal, test


def target_cases(cfg):
    """Cases to check: (prob, label or None, image or None)."""
    probs, labels, images = _index(cfg, "target")
    out = {}
    for c, p in sorted(probs.items()):
        out[c] = (p, (labels or {}).get(c), (images or {}).get(c))
    if labels is not None:
        n = sum(1 for v in out.values() if v[1])
        log.info("%d target cases, %d with reference labels", len(out), n)
    return out


def load_case(case_id, prob_path, label_path, cfg, expect=None, image_path=None, need_geometry=False):
    """Load one case: probabilities (channels first, label order), optional label, spacing and geometry."""
    inp = cfg["input"]
    arr, meta = io.load_prob_file(prob_path, inp["format"])
    lab = lab_img = spacing = units = None
    if label_path:
        lab, lab_img, spacing, units = io.load_label(label_path, inp["binarize_255"])
    arr, has_ch, transposed = io.standardize(
        arr, meta, None if lab is None else lab.shape, None if lab is not None else expect, inp["array_order"])
    # target cases reuse the input type detected at calibration (eg, logits), so both are read identically
    requested = expect["raw_kind"] if expect else inp["prob_kind"]
    arr, kind, raw_kind = io.to_probabilities(arr, has_ch, requested)
    case = {
        "id": case_id, "probs": arr, "has_ch": has_ch, "kind": kind, "raw_kind": raw_kind, "transposed": transposed,
        "label": lab, "spacing": spacing, "units": units, "meta": meta,
    }
    if need_geometry:
        case["geometry"] = io.geometry_reference(case_id, meta, lab_img, image_path)
    if spacing is None and need_geometry and case["geometry"] is not None:
        g = case["geometry"]
        sp = g["spacing"] if isinstance(g, dict) else g.GetSpacing()
        case["spacing"] = tuple(float(s) for s in reversed(sp))
        case["units"] = "mm"
    return case
