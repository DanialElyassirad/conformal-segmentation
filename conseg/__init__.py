"""conseg: conformal prediction-based uncertainty for segmentation model outputs.

Python API
----------
    from conseg import load_config, calibrate, predict, evaluate, run_all
    cfg = load_config("config.yaml")
    calib, cases, review, summary = run_all(cfg)
"""
__version__ = "0.1.0"

from .config import load_config  # noqa: E402,F401
from .calibrate import calibrate, load_calibration, save_calibration  # noqa: E402,F401
from .predict import predict  # noqa: E402,F401
from .evaluate import evaluate  # noqa: E402,F401
from .pipeline import run_all  # noqa: E402,F401
