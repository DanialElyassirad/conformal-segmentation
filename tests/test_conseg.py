import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from conseg import core, io, metrics
from conseg.cli import main
from conseg.config import load_config
from conseg.pipeline import run_all, setup_logging

import synth


def _cfg(tmp, data, **over):
    base = {
        "labels": {"organ": [1], "lesion": [2], "whole": [1, 2]},
        "calibration": {"probabilities": os.path.join(data, "probs"), "labels": os.path.join(data, "labels"),
                        "split": 0.4, "nbins": 200000},
        "evaluation": {"bootstrap": 200, "permutations": 500},
        "outputs": {"dir": os.path.join(tmp, "out")},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k != "labels":
            base[k].update(v)
        else:
            base[k] = v
    return load_config(None, base)


# ----------------------------------------------------------------------------- unit tests
def test_dice_per_threshold_matches_bruteforce():
    rng = np.random.default_rng(1)
    p = rng.random((10, 20, 20)).astype(np.float32)
    g = rng.random((10, 20, 20)) > 0.7
    fast = core.dice_per_threshold(p, g)
    slow = np.array([core.dice(p > t, g) for t in core.THRESHOLDS])
    assert np.allclose(fast, slow)


def test_streaming_quantile_is_conservative_and_close():
    rng = np.random.default_rng(2)
    s = rng.beta(0.5, 3, 200000)
    sq = core.StreamingQuantile(100000)
    sq.add(s[:100000]); sq.add(s[100000:])
    for a in (0.1, 0.01, 0.001):
        k = int(np.ceil((len(s) + 1) * (1 - a)))
        exact = np.sort(s)[k - 1]
        q = sq.conformal_threshold(a)
        assert exact <= q <= exact + 2e-5


def test_normalize_maps_threshold_to_half():
    p = np.array([0.0, 0.02, 0.06, 0.5, 1.0], np.float32)
    q = core.normalize(p, 0.06)
    assert q[0] == pytest.approx(0.0) and q[2] == pytest.approx(0.5, abs=1e-5) and q[-1] == pytest.approx(1.0)
    assert np.all(np.diff(q) >= 0)


def test_hd95_known_offset():
    a = np.zeros((20, 30, 30), bool); a[5:15, 5:15, 5:15] = True
    b = np.roll(a, 3, axis=2)
    spacing = (2.5, 0.8, 0.8)
    assert metrics.hd95(a, b, spacing) == pytest.approx(3 * 0.8, abs=1e-6)
    assert np.isnan(metrics.hd95(a, np.zeros_like(a), spacing))
    assert metrics.hd95(a, np.zeros_like(a), spacing, empty="diagonal") > 0


def test_to_probabilities_detects_logits_and_softmax():
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 3, (3, 4, 5, 5)).astype(np.float32)
    p, kind, raw = io.to_probabilities(logits, True)
    assert raw == "logits_softmax" and kind == "softmax" and np.allclose(p.sum(0), 1, atol=1e-5)
    p2, kind2, raw2 = io.to_probabilities(p, True)
    assert raw2 == "softmax" and kind2 == "softmax"


# ----------------------------------------------------------------------------- end-to-end
@pytest.fixture(scope="module")
def nnunet_data(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("nnunet"))
    synth.make_dataset(root, n=40, seed=0, layout="nnunet_v2")
    return root


def test_end_to_end_split(tmp_path, nnunet_data):
    cfg = _cfg(str(tmp_path), nnunet_data)
    setup_logging(None)
    calib, df, review, summary = run_all(cfg)
    out = cfg["outputs"]["dir"]
    for f in ("calibration.json", "cases.csv", "review_list.csv", "config_used.yaml", "run_summary.json",
              "evaluation/summary.csv", "evaluation/retention.csv", "evaluation/risk_coverage_organ.png"):
        assert os.path.exists(os.path.join(out, f)), f
    # maps written as NIfTI with the label geometry (from CASE.nii.gz next to the .npz)
    m = df["conformal_map"].iloc[0]
    assert m.endswith(".nii.gz") and os.path.exists(m)
    # one small split: coverage near the target (the guarantee holds on average; see test_average_coverage)
    for name, t in calib["targets"].items():
        d = df[df.target == name]
        cov = (d.coverage * d.n_voxels).sum() / d.n_voxels.sum()
        assert cov >= t["target_coverage"] - 5e-3, (name, cov, t["target_coverage"])
        assert 0.01 < t["ncst"] < 0.99
    # uncertainty ratio tracks segmentation quality
    d = df[df.target == "whole"].dropna(subset=["UR"])
    rho = stats.spearmanr(d.UR, d.dice_baseline).correlation
    assert rho < -0.3, rho
    # top-20% rule flags ceil(0.2 n) cases per target (plus undefined ones)
    n = df.case.nunique()
    assert review.flagged.sum() >= int(np.ceil(0.2 * n))
    s = summary.set_index(["target", "metric"])
    assert s.loc[("whole", "DSC"), "AURCC_UR"] < s.loc[("whole", "DSC"), "AURCC_random"]


def test_average_coverage_meets_target(tmp_path):
    """Averaged over random calibration/test splits, pooled voxel coverage is close to 1 - alpha.
    Voxels within a case are correlated (not exchangeable), so the voxel-level guarantee is approximate:
    a shortfall of a few tenths of a percentage point is expected."""
    from conseg.calibrate import calibrate
    from conseg.data import calibration_pool
    from conseg.predict import predict
    data = str(tmp_path / "d")
    synth.make_dataset(data, n=60, seed=11, layout="nnunet_v2", shape=(8, 24, 24))
    setup_logging(None)
    covs = []
    for seed in range(5):
        cfg = _cfg(str(tmp_path), data, labels={"organ": [1]},
                   calibration={"split": 0.5, "seed": seed, "alpha": 0.05},
                   outputs={"dir": str(tmp_path / f"o{seed}"), "write_maps": False, "write_baseline": False},
                   evaluation={"hd95": False})
        cal, test = calibration_pool(cfg)
        calib = calibrate(cfg, cal)
        df, _ = predict(cfg, calib, {c: (p, l, None) for c, (p, l) in test.items()}, str(tmp_path / f"o{seed}"))
        covs.append((df.coverage * df.n_voxels).sum() / df.n_voxels.sum())
    assert np.mean(covs) >= 0.95 - 5e-3, covs


def test_separate_target_folder_and_cli(tmp_path, nnunet_data):
    # calibrate on half the cases (held-out mode), predict the other half without labels via the CLI
    import shutil
    cal, new = tmp_path / "cal", tmp_path / "new"
    for d in (cal / "probs", cal / "labels", new):
        d.mkdir(parents=True)
    files = sorted(os.listdir(os.path.join(nnunet_data, "labels")))
    for i, f in enumerate(files):
        cid = f[:-7]
        dst = cal if i < 20 else None
        if dst:
            shutil.copy(os.path.join(nnunet_data, "probs", cid + ".npz"), dst / "probs")
            shutil.copy(os.path.join(nnunet_data, "labels", f), dst / "labels")
        else:
            shutil.copy(os.path.join(nnunet_data, "probs", cid + ".npz"), new)
            shutil.copy(os.path.join(nnunet_data, "probs", cid + ".nii.gz"), new)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "labels: {organ: [1]}\n"
        f"calibration: {{probabilities: '{(cal / 'probs').as_posix()}', labels: '{(cal / 'labels').as_posix()}', nbins: 200000}}\n"
        f"target: {{probabilities: '{new.as_posix()}'}}\n"
        "flagging: {rule: calibration_percentile, value: 90}\n"
        f"outputs: {{dir: '{(tmp_path / 'out').as_posix()}'}}\n", encoding="utf-8")
    main(["run", "-c", str(cfg_path)])
    rev = pd.read_csv(tmp_path / "out" / "review_list.csv")
    assert len(rev) == 20 and "flagged" in rev
    calib = json.loads((tmp_path / "out" / "calibration.json").read_text())
    assert calib["input"]["raw_kind"] == "softmax" and len(calib["targets"]["organ"]["ur_percentiles"]) == 99
    assert not (tmp_path / "out" / "evaluation").exists()          # no labels -> no evaluation


def test_logits_input(tmp_path):
    synth.make_dataset(str(tmp_path / "d"), n=16, seed=1, layout="logits")
    cfg = _cfg(str(tmp_path), str(tmp_path / "d"), labels={"organ": [1]})
    setup_logging(None)
    calib, df, _, _ = run_all(cfg)
    assert calib["input"]["raw_kind"] == "logits_softmax"
    assert df["dice_baseline"].mean() > 0.3


def test_xyz_order_is_transposed(tmp_path):
    synth.make_dataset(str(tmp_path / "d"), n=12, seed=2, layout="xyz")
    cfg = _cfg(str(tmp_path), str(tmp_path / "d"), labels={"organ": [1]})
    setup_logging(None)
    calib, df, _, _ = run_all(cfg)
    assert calib["input"]["transposed"] is True
    assert df["dice_baseline"].mean() > 0.3


def test_single_channel_composite(tmp_path):
    synth.make_dataset(str(tmp_path / "d"), n=12, seed=3, layout="single_channel")
    cfg = _cfg(str(tmp_path), str(tmp_path / "d"), labels={"organ": [1, 2]})
    setup_logging(None)
    calib, df, _, _ = run_all(cfg)
    assert calib["input"]["has_channels"] is False
    assert df["dice_baseline"].mean() > 0.3


def test_single_channel_rejects_multiple_targets(tmp_path):
    synth.make_dataset(str(tmp_path / "d"), n=6, seed=4, layout="single_channel")
    cfg = _cfg(str(tmp_path), str(tmp_path / "d"))
    setup_logging(None)
    with pytest.raises(ValueError, match="exactly one"):
        run_all(cfg)


def test_nnunet_v1_cropped_softmax(tmp_path):
    rng = np.random.default_rng(5)
    pdir, ldir = tmp_path / "probs", tmp_path / "labels"
    pdir.mkdir(); ldir.mkdir()
    for i in range(10):
        label, probs, _ = synth.make_case(rng, (16, 40, 40), 0.3)
        bbox = [[2, 14], [4, 36], [5, 35]]
        crop = probs[:, 2:14, 4:36, 5:35]
        cid = f"v1_{i:02d}"
        np.savez_compressed(pdir / f"{cid}.npz", softmax=crop.astype(np.float16))
        with open(pdir / f"{cid}.pkl", "wb") as f:
            pickle.dump({"crop_bbox": bbox, "original_size_of_raw_data": (16, 40, 40)}, f)
        synth.write_image(label, str(ldir / f"{cid}.nii.gz"))
    cfg = load_config(None, {"labels": {"organ": [1]},
                             "calibration": {"probabilities": str(pdir), "labels": str(ldir), "split": 0.5,
                                             "nbins": 100000},
                             "evaluation": {"bootstrap": 50, "permutations": 100},
                             "outputs": {"dir": str(tmp_path / "out")}})
    setup_logging(None)
    calib, df, _, _ = run_all(cfg)
    assert df["dice_baseline"].mean() > 0.5


def test_2d_png_labels(tmp_path):
    import SimpleITK as sitk
    rng = np.random.default_rng(6)
    pdir, ldir = tmp_path / "p", tmp_path / "l"
    pdir.mkdir(); ldir.mkdir()
    for i in range(20):
        label, probs, _ = synth.make_case(rng, (3, 64, 64), float(rng.uniform(0.1, 1)))
        lab2d = (label[1] > 0).astype(np.uint8) * 255
        sitk.WriteImage(sitk.GetImageFromArray(lab2d), str(ldir / f"img_{i:02d}.png"))
        np.save(pdir / f"img_{i:02d}.npy", (probs[1] + probs[2])[1])
    cfg = load_config(None, {"labels": {"lesion": [1]}, "input": {"label_suffix": ".png"},
                             "calibration": {"probabilities": str(pdir), "labels": str(ldir), "split": 0.4,
                                             "nbins": 100000},
                             "evaluation": {"bootstrap": 50, "permutations": 100},
                             "outputs": {"dir": str(tmp_path / "out")}})
    setup_logging(None)
    calib, df, _, summary = run_all(cfg)
    assert calib["input"]["units"] == "px"
    assert set(summary.metric) == {"DSC", "HD95 (px)"}


def test_nnunet_2d_natural_image_layout(tmp_path):
    # nnU-Net v2 saves 2D natural-image probabilities as C x 1 x H x W next to H x W PNG labels
    import SimpleITK as sitk
    rng = np.random.default_rng(7)
    pdir, ldir, new = tmp_path / "p", tmp_path / "l", tmp_path / "new"
    for d in (pdir, ldir, new):
        d.mkdir()
    for i in range(16):
        label, probs, _ = synth.make_case(rng, (3, 64, 64), float(rng.uniform(0.1, 1)))
        p2 = probs[:, 1:2]                                   # C x 1 x H x W
        lab2d = (label[1] > 0).astype(np.uint8)
        p2 = np.stack([p2[0], p2[1] + p2[2]])
        sitk.WriteImage(sitk.GetImageFromArray(lab2d), str(ldir / f"img_{i:02d}.png"))
        np.savez_compressed((pdir if i < 12 else new) / f"img_{i:02d}.npz", probabilities=p2.astype(np.float16))
    cfg = load_config(None, {"labels": {"lesion": [1]}, "input": {"label_suffix": ".png"},
                             "calibration": {"probabilities": str(pdir), "labels": str(ldir), "nbins": 100000},
                             "target": {"probabilities": str(new)},
                             "outputs": {"dir": str(tmp_path / "out")}})
    setup_logging(None)
    calib, df, _, _ = run_all(cfg)
    assert calib["input"]["spatial_ndim"] == 2 and len(df) == 4


def test_region_based_channels(tmp_path):
    # sigmoid channel 0 = whole (labels 1, 2), channel 1 = core (label 2); no background channel
    rng = np.random.default_rng(8)
    pdir, ldir = tmp_path / "p", tmp_path / "l"
    pdir.mkdir(); ldir.mkdir()
    for i in range(16):
        label, probs, _ = synth.make_case(rng, (16, 40, 40), float(rng.uniform(0.1, 1)))
        regions = np.stack([probs[1] + probs[2], probs[2]])
        np.savez_compressed(pdir / f"r_{i:02d}.npz", probabilities=regions.astype(np.float16))
        synth.write_image(label, str(ldir / f"r_{i:02d}.nii.gz"))
    base = {"labels": {"whole": [1, 2], "core": [2]},
            "calibration": {"probabilities": str(pdir), "labels": str(ldir), "split": 0.5, "nbins": 100000},
            "evaluation": {"bootstrap": 50, "permutations": 100}, "outputs": {"dir": str(tmp_path / "out")}}
    setup_logging(None)
    with pytest.raises(ValueError, match="set 'channels'"):
        run_all(load_config(None, base))
    calib, df, _, _ = run_all(load_config(None, {**base, "channels": {"whole": 0, "core": 1}}))
    assert calib["input"]["kind"] == "sigmoid" and calib["targets"]["core"]["channels"] == [1]
    assert df.groupby("target").dice_baseline.mean().min() > 0.3
    with pytest.raises(ValueError, match="channels.nope"):
        load_config(None, {**base, "channels": {"nope": 0}})


def test_flagging_ignores_absent_targets():
    from conseg.predict import flag_cases
    cfg = load_config(None, {"flagging": {"rule": "top_fraction", "value": 0.5}})
    df = pd.DataFrame({"case": list("abcdef"), "target": "lesion",
                       "UR": [np.nan, np.nan, np.nan, 0.1, 2.0, 1.0],
                       "n_uncertain": [0, 0, 7, 3, 40, 20]})
    df, review = flag_cases(df, {"targets": {}}, cfg)
    flagged = set(review[review.flagged].case)
    # a, b: nothing detected -> not flagged and no slot used; c: uncertain only -> flagged;
    # top 50% of the 3 cases with a UR -> e and f
    assert flagged == {"c", "e", "f"}
    assert df.set_index("case").loc["a", "reason"].startswith("nothing detected")


def test_bad_config_is_reported():
    with pytest.raises(ValueError, match="flagging.rule"):
        load_config(None, {"flagging": {"rule": "nonsense"}})
