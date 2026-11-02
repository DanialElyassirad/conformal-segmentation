"""High-level workflow: calibrate -> predict -> evaluate."""
import json
import logging
import os

import yaml

from . import __version__
from .calibrate import calibrate, load_calibration, save_calibration
from .config import resolve_paths
from .data import target_cases
from .evaluate import evaluate
from .predict import predict

log = logging.getLogger("conseg")


def setup_logging(outdir=None, verbose=False):
    log.handlers.clear()
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); log.addHandler(sh)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(outdir, "run.log"), encoding="utf-8")
        fh.setFormatter(fmt); log.addHandler(fh)


def output_dir(cfg):
    out = resolve_paths(cfg, cfg["outputs"]["dir"])[0]
    os.makedirs(out, exist_ok=True)
    return out


def _snapshot(cfg, outdir):
    snap = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(os.path.join(outdir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"conseg_version": __version__, **snap}, f, sort_keys=False)


def run_calibrate(cfg):
    out = output_dir(cfg)
    calib = calibrate(cfg)
    save_calibration(calib, os.path.join(out, "calibration.json"))
    log.info("Calibration saved -> %s", os.path.join(out, "calibration.json"))
    return calib


def _cases_to_check(cfg, calib):
    if cfg["target"]["probabilities"]:
        return target_cases(cfg)
    tp = calib.get("_test_pairs") or {}
    if tp:
        log.info("No target folder configured; using the %d held-out labeled cases from calibration.split", len(tp))
        return {c: (p, l, None) for c, (p, l) in tp.items()}
    raise ValueError("Nothing to predict: set target.probabilities, or calibration.split to hold out test cases")


def run_predict(cfg, calib=None, calib_path=None):
    out = output_dir(cfg)
    if calib is None:
        calib = load_calibration(calib_path or os.path.join(out, "calibration.json"))
    cases = _cases_to_check(cfg, calib)
    return predict(cfg, calib, cases, out)


def run_all(cfg):
    out = output_dir(cfg)
    _snapshot(cfg, out)
    calib = run_calibrate(cfg)
    df, review = run_predict(cfg, calib)
    summary = evaluate(cfg, calib, df, os.path.join(out, "evaluation"))
    with open(os.path.join(out, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n_cases": int(df["case"].nunique()), "n_flagged": int(review["flagged"].sum()),
                   "targets": list(calib["targets"]), "evaluated": summary is not None}, f, indent=2)
    return calib, df, review, summary
