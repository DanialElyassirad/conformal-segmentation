"""Configuration: defaults, YAML loading and validation."""
import copy
import os

import yaml

DEFAULTS = {
    # name -> list of label values in the reference masks (and channel indices in a softmax output).
    # A list with several values defines a composite target, e.g. whole_pancreas: [1, 2].
    # None = one target per foreground label value found in the calibration labels.
    "labels": None,
    # name -> probability channel indices, for targets whose channels are not their label values
    # (eg, region-based nnU-Net: channel 0 = first region). None/missing = channels equal label values.
    "channels": None,
    "input": {
        "format": "auto",        # auto | nnunet_v2 | nnunet_v1 | npz | npy | image
        "prob_kind": "auto",     # auto | softmax | sigmoid | logits_softmax | logits_sigmoid
        "array_order": "auto",   # auto | zyx | xyz  (zyx = SimpleITK/nnU-Net order)
        "prob_suffix": None,     # e.g. ".npz"; None = choose automatically
        "label_suffix": None,    # e.g. ".nii.gz"; None = any supported image file
        "binarize_255": True,    # map {0, 255} label images (PNG) to {0, 1}
    },
    "calibration": {
        "probabilities": None,   # folder(s) or glob(s) with calibration probability maps
        "labels": None,          # folder(s) with the matching reference masks
        "split": None,           # e.g. 0.3: use 30% of the labeled pool for calibration, rest for testing
        "seed": 42,
        "patient_map": None,     # optional CSV (case,patient) so one patient never spans both sets
        "threshold": "auto",     # auto = Dice-optimal threshold (BMOT); or a number, e.g. 0.5
        "alpha": "auto",         # auto, a number, or {target_name: number}
        "alpha_ratio": 0.28,     # auto alpha starts at alpha_ratio x pooled foreground fraction
        "nbins": 1000000,        # resolution of the streaming quantile
    },
    "target": {
        "probabilities": None,   # folder(s) with probability maps of the cases to check
        "labels": None,          # optional reference masks (enables evaluation)
        "images": None,          # optional images (geometry for output maps)
    },
    "flagging": {
        "rule": "top_fraction",  # top_fraction | ur_threshold | calibration_percentile
        "value": 0.2,            # fraction (0-1), UR value, or percentile (0-100)
        "flag_undefined": True,  # flag cases with uncertain but no certain-foreground voxels
    },
    "outputs": {
        "dir": "conseg_output",
        "write_maps": True,      # conformal maps (0 certain background, 1 uncertain, 2 certain foreground)
        "write_baseline": True,  # baseline masks thresholded at the BMOT
    },
    "evaluation": {
        "hd95": True,
        "empty_hd95": "nan",     # nan | diagonal  (HD95 when a mask is empty)
        "bootstrap": 1000,
        "permutations": 10000,
        "retention": [0.9, 0.5, 0.2],
        "seed": 42,
    },
    "workers": 1,
}

FLAG_RULES = ("top_fraction", "ur_threshold", "calibration_percentile")
PROB_KINDS = ("auto", "softmax", "sigmoid", "logits_softmax", "logits_sigmoid")
FORMATS = ("auto", "nnunet_v2", "nnunet_v1", "npz", "npy", "image")


def _merge(base, new):
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides=None):
    """Load a YAML config (optional), merge it over the defaults and validate it."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = _merge(cfg, user)
        cfg["_config_dir"] = os.path.dirname(os.path.abspath(path))
    if overrides:
        cfg = _merge(cfg, overrides)
    validate(cfg)
    return cfg


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def validate(cfg):
    errors = []
    if cfg["input"]["format"] not in FORMATS:
        errors.append(f"input.format must be one of {FORMATS}")
    if cfg["input"]["prob_kind"] not in PROB_KINDS:
        errors.append(f"input.prob_kind must be one of {PROB_KINDS}")
    if cfg["input"]["array_order"] not in ("auto", "zyx", "xyz"):
        errors.append("input.array_order must be auto, zyx or xyz")
    labels = cfg.get("labels")
    if labels is not None:
        if not isinstance(labels, dict) or not labels:
            errors.append("labels must be a mapping name -> list of label values, e.g. {pancreas: [1]}")
        else:
            for name, vals in labels.items():
                vals = _as_list(vals)
                if not vals or not all(isinstance(v, int) and v > 0 for v in vals):
                    errors.append(f"labels.{name} must be a positive integer or a list of them")
                labels[name] = vals
    chans = cfg.get("channels")
    if chans is not None:
        if not isinstance(chans, dict) or not labels:
            errors.append("channels must be a mapping name -> channel indices, and needs 'labels' to be set")
        else:
            for name, vals in chans.items():
                vals = _as_list(vals)
                if name not in labels:
                    errors.append(f"channels.{name} has no matching entry in 'labels'")
                if not vals or not all(isinstance(v, int) and v >= 0 for v in vals):
                    errors.append(f"channels.{name} must be a non-negative integer or a list of them")
                chans[name] = vals
    c = cfg["calibration"]
    if c["split"] is not None and not (0 < float(c["split"]) < 1):
        errors.append("calibration.split must be between 0 and 1")
    if c["threshold"] != "auto":
        try:
            t = float(c["threshold"])
            assert 0 < t < 1
        except Exception:
            errors.append("calibration.threshold must be 'auto' or a number between 0 and 1")
    a = c["alpha"]
    if a != "auto" and not isinstance(a, dict):
        try:
            assert 0 < float(a) < 1
        except Exception:
            errors.append("calibration.alpha must be 'auto', a number between 0 and 1, or {target: number}")
    fl = cfg["flagging"]
    if fl["rule"] not in FLAG_RULES:
        errors.append(f"flagging.rule must be one of {FLAG_RULES}")
    if cfg["evaluation"]["empty_hd95"] not in ("nan", "diagonal"):
        errors.append("evaluation.empty_hd95 must be nan or diagonal")
    if errors:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
    return cfg


def resolve_paths(cfg, value):
    """Resolve relative paths against the config file folder; keep lists and globs."""
    base = cfg.get("_config_dir", os.getcwd())
    out = []
    for p in _as_list(value):
        p = os.path.expanduser(str(p))
        out.append(p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p)))
    return out
