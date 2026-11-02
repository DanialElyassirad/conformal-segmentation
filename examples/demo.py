"""Self-contained demo: creates a small synthetic nnU-Net-style dataset and runs the full pipeline.

    python examples/demo.py

Outputs go to ./demo_data/ (data) and ./conseg_output/ (results). Requires no real data.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
sys.path.insert(0, os.path.join(HERE, ".."))

import synth  # noqa: E402
from conseg import load_config, run_all  # noqa: E402
from conseg.pipeline import setup_logging  # noqa: E402

data = os.path.abspath("demo_data")
print("Creating synthetic cases in", data)
synth.make_dataset(data, n=40, seed=0, layout="nnunet_v2")

cfg = load_config(None, {
    "labels": {"organ": [1], "lesion": [2], "whole": [1, 2]},
    "calibration": {"probabilities": os.path.join(data, "probs"), "labels": os.path.join(data, "labels"),
                    "split": 0.3},
    "flagging": {"rule": "top_fraction", "value": 0.2},
    "outputs": {"dir": os.path.abspath("conseg_output")},
})
setup_logging(cfg["outputs"]["dir"])
calib, cases, review, summary = run_all(cfg)
print("\nReview list (top 10):")
print(review.head(10).to_string(index=False))
print("\nEvaluation summary:")
print(summary[["target", "metric", "AURCC_UR", "AURCC_random", "P_two_sided", "pooled_coverage",
               "target_coverage"]].to_string(index=False))
