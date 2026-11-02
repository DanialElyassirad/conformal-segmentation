"""Command-line interface:  conseg {run, calibrate, predict, evaluate, inspect, init-config}"""
import argparse
import os
import sys
from importlib import resources

import numpy as np

from . import __version__
from .config import load_config


def _cfg(args):
    over = {}
    if getattr(args, "out", None):
        over["outputs"] = {"dir": os.path.abspath(args.out)}
    if getattr(args, "workers", None):
        over["workers"] = args.workers
    return load_config(args.config, over)


def cmd_run(args):
    from .pipeline import output_dir, run_all, setup_logging
    cfg = _cfg(args); setup_logging(output_dir(cfg), args.verbose)
    run_all(cfg)


def cmd_calibrate(args):
    from .pipeline import output_dir, run_calibrate, setup_logging
    cfg = _cfg(args); setup_logging(output_dir(cfg), args.verbose)
    run_calibrate(cfg)


def cmd_predict(args):
    from .pipeline import output_dir, run_predict, setup_logging
    cfg = _cfg(args); setup_logging(output_dir(cfg), args.verbose)
    run_predict(cfg, calib_path=args.calibration)


def cmd_evaluate(args):
    import pandas as pd
    from .calibrate import load_calibration
    from .evaluate import evaluate
    from .pipeline import output_dir, setup_logging
    cfg = _cfg(args); out = output_dir(cfg); setup_logging(out, args.verbose)
    calib = load_calibration(args.calibration or os.path.join(out, "calibration.json"))
    df = pd.read_csv(os.path.join(out, "cases.csv"))
    evaluate(cfg, calib, df, os.path.join(out, "evaluation"))


def cmd_inspect(args):
    from . import io
    arr, meta = io.load_prob_file(args.path)
    print(f"file:        {args.path}")
    print(f"format:      {meta['format']}" + (f" (key '{meta.get('key')}')" if meta.get("key") else ""))
    print(f"shape:       {arr.shape}  dtype(read as): float32")
    print(f"value range: {arr.min():.4g} .. {arr.max():.4g}")
    if arr.ndim >= 3 and arr.shape[0] <= 32:
        s = arr.reshape(arr.shape[0], -1)
        idx = np.linspace(0, s.shape[1] - 1, min(5000, s.shape[1])).astype(int)
        print(f"channels?:   first axis has {arr.shape[0]} entries; mean channel sum = {s[:, idx].sum(0).mean():.3f} "
              f"(~1 means softmax)")
    if meta.get("pkl"):
        print(f"properties:  {meta['pkl']}")
    lo, hi = float(arr.min()), float(arr.max())
    if lo < -1e-4 or hi > 1 + 1e-3:
        print("note:        values outside [0, 1] -> will be treated as logits")


def cmd_init(args):
    name = args.template
    try:
        text = resources.files("conseg.templates").joinpath(f"{name}.yaml").read_text(encoding="utf-8")
    except FileNotFoundError:
        avail = [p.name[:-5] for p in resources.files("conseg.templates").iterdir() if p.name.endswith(".yaml")]
        sys.exit(f"Unknown template '{name}'. Available: {', '.join(sorted(avail))}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {args.output}")
    else:
        print(text)


def main(argv=None):
    p = argparse.ArgumentParser(prog="conseg", description="Conformal uncertainty for segmentation model outputs")
    p.add_argument("--version", action="version", version=f"conseg {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, calib=False):
        sp.add_argument("-c", "--config", required=True, help="YAML configuration file")
        sp.add_argument("-o", "--out", help="output folder (overrides outputs.dir)")
        sp.add_argument("-w", "--workers", type=int, help="parallel worker processes")
        sp.add_argument("-v", "--verbose", action="store_true")
        if calib:
            sp.add_argument("--calibration", help="calibration.json (default: <out>/calibration.json)")

    common(sub.add_parser("run", help="calibrate, predict and (if labels exist) evaluate"))
    common(sub.add_parser("calibrate", help="calibrate on labeled cases"))
    common(sub.add_parser("predict", help="apply a calibration to new cases"), calib=True)
    common(sub.add_parser("evaluate", help="evaluate predictions that have reference labels"), calib=True)
    sp = sub.add_parser("inspect", help="show what is inside a probability file")
    sp.add_argument("path")
    sp = sub.add_parser("init-config", help="write a configuration template")
    sp.add_argument("template", help="nnunet_v2_oof | nnunet_v2_heldout | split_pool | generic_2d")
    sp.add_argument("-o", "--output", help="file to write (default: print)")

    args = p.parse_args(argv)
    {"run": cmd_run, "calibrate": cmd_calibrate, "predict": cmd_predict, "evaluate": cmd_evaluate,
     "inspect": cmd_inspect, "init-config": cmd_init}[args.cmd](args)


if __name__ == "__main__":   # pragma: no cover
    main()
