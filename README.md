# conseg — conformal uncertainty for segmentation model outputs

`conseg` turns the softmax/probability output of a segmentation model (nnU-Net or any other) into

* **voxel-level uncertainty maps** — every voxel is *certain foreground*, *certain background*, or *uncertain*;
* a **case-level uncertainty ratio (UR)** — uncertain voxels / certain-foreground voxels;
* a **ranked review list** that flags cases most likely to need manual revision;
* optionally, an **evaluation report** (coverage, Dice, HD95, risk–coverage analysis) when reference masks are available.

It is post hoc: the model is **not retrained** and its weights are not needed — only its probability maps and a set of labeled calibration cases.

> Research software, not a medical device.

---

## Contents
1. [Installation](#installation)
2. [Quick start with nnU-Net](#quick-start-with-nnu-net)
3. [How it works](#how-it-works)
4. [Which calibration data to use](#which-calibration-data-to-use)
5. [Supported inputs](#supported-inputs)
6. [Configuration reference](#configuration-reference)
7. [Outputs](#outputs)
8. [Flagging cases for review](#flagging-cases-for-review)
9. [Command line and Python API](#command-line-and-python-api)
10. [Troubleshooting](#troubleshooting)
11. [Limitations](#limitations)

---

## Installation

```bash
git clone <repository-url> conformal-segmentation
cd conformal-segmentation
pip install .            # installs the `conseg` command
pip install .[test]      # optional: run the tests with `pytest`
```

Python ≥ 3.9. Dependencies: numpy, scipy, SimpleITK, pandas, PyYAML, matplotlib. CPU only; no internet access is needed after installation.

Try it without any data:

```bash
python examples/demo.py      # builds a small synthetic dataset and runs the whole pipeline
```

## Quick start with nnU-Net

**1. Make sure probabilities are saved.** `conseg` needs the softmax probabilities, not only the final masks.

```bash
# calibration data: out-of-fold predictions of the training cases (one command per fold)
nnUNetv2_train DATASET_ID 3d_fullres FOLD --val --npz
# new cases to check
nnUNetv2_predict -i IMAGES -o PREDICTIONS -d DATASET_ID -c 3d_fullres --save_probabilities
```

**2. Write a config** (templates are included):

```bash
conseg init-config nnunet_v2_oof -o config.yaml
# edit the paths and the label names/values in config.yaml
```

Replace `nnunet_v2_oof` with the template that matches your scenario (see [Which calibration data to use](#which-calibration-data-to-use)):

| Template | Scenario |
|---|---|
| `nnunet_v2_oof` | nnU-Net trained with 5-fold cross-validation; calibrate on the out-of-fold predictions |
| `nnunet_v2_heldout` | a separate labeled set that was not used for training |
| `split_pool` | one labeled pool, split into calibration and test sets (full evaluation report) |
| `generic_2d` | any other model, 2D images (eg, PNG masks) |

**3. Run:**

```bash
conseg run -c config.yaml
```

Results appear in `conseg_output/`; start with `review_list.csv`.

To check what is inside a probability file: `conseg inspect PREDICTIONS/case_001.npz`.

## How it works

For each target structure:

1. **Threshold.** On the calibration cases, the threshold that maximizes the mean Dice (the *base model optimal threshold*, BMOT) is found (or a fixed threshold is used).
2. **Normalization.** Each case's probability map *p* is mapped piecewise-linearly so that the BMOT becomes 0.5 (case minimum → 0, maximum → 1). This keeps conformal decisions consistent with the baseline mask.
3. **Calibration.** For every calibration voxel, the nonconformity score is 1 minus the normalized score of its true class. The nonconformity score threshold (NCST) is the ⌈(n + 1)(1 − α)⌉/n empirical quantile of all pooled scores (*n* = number of calibration voxels, α = miscoverage rate).
4. **Prediction sets.** For a new voxel, *foreground* is in the set if 1 − q ≤ NCST and *background* if q ≤ NCST. One class → certain; both or neither → uncertain.
5. **Uncertainty ratio.** UR = uncertain voxels / certain-foreground voxels. High UR = a large part of the predicted structure is uncertain. UR is undefined when no voxel is certain foreground. Such cases are flagged by default if they contain uncertain voxels; cases in which every voxel is certain background (nothing detected, eg, a lesion absent from the case) are not flagged.
6. **Evaluation (optional).** Cases are excluded from most to least uncertain; the mean Dice/HD95 of the retained cases is plotted against the fraction retained (risk–coverage curve). The area under this curve (AURCC) is compared with random exclusion (bootstrap CIs, permutation test).

**Multi-class models.** Each target is calibrated separately (one-vs-rest), with its own threshold, α, and NCST. Composite targets are supported, e.g. `whole_pancreas: [1, 2]` uses the sum of the softmax channels 1 and 2.

**Region-based models** (eg, nnU-Net trained with `regions_class_order`) output one sigmoid channel per region, with no background channel, so channel *k* is not label value *k*. Give the label values of each region in `labels` and its channel in `channels`. For example, with masks labeled 1 = pancreas and 2 = tumor, and regions trained as whole pancreas (1 + 2) and tumor (2):

```yaml
labels:   {whole_pancreas: [1, 2], tumor: [2]}
channels: {whole_pancreas: 0, tumor: 1}
```

**Choosing α.** `alpha: auto` starts at 0.28 × the target's foreground fraction and adjusts until the NCST is non-degenerate (between 0.01 and 0.99). Small structures need small α. For a strict a priori guarantee, fix α in the config instead.

## Which calibration data to use

**Calibration cases must not have been used to train the model.** Using training-set predictions would make the model look over-confident and the uncertainty threshold too permissive. In particular, do not run the final model (or the 5-fold ensemble) on its own training images and calibrate on that output; use the out-of-fold predictions below.

**Single fold vs ensemble.** Out-of-fold predictions each come from one fold's model, whereas `nnUNetv2_predict` averages all five folds by default. Averaged probabilities are smoother, so new cases tend to receive somewhat higher UR than calibration cases. Ranking (`flagging.rule: top_fraction`) is unaffected; for `calibration_percentile`, a fixed `ur_threshold`, or a strict coverage claim, predict new cases with a single fold (`-f 0`) or calibrate on a held-out set predicted by the same ensemble.

| Situation | Mode | Config |
|---|---|---|
| nnU-Net trained with 5-fold cross-validation | **Out-of-fold** (recommended): use `fold_*/validation` predictions of the training cases | `calibration.probabilities: .../fold_*/validation` |
| Separate labeled held-out set | **Held-out** | point `calibration.*` to it |
| One labeled pool, no model training on it | **Split** | `calibration.split: 0.3` |

Calibration data should come from the same scanners, protocols, and population as the cases you want to check. When the model is used at a new site, recalibrate on local labeled cases (only `conseg calibrate` needs to be re-run).

## Supported inputs

**Probabilities** (one file per case; case ID = file name without extension):

| Source | File | Notes |
|---|---|---|
| nnU-Net v2 | `CASE.npz` (key `probabilities`, C × Z × Y × X) | geometry from `CASE.nii.gz` or `CASE.pkl` next to it |
| nnU-Net v1 | `CASE.npz` (key `softmax`) + `CASE.pkl` | cropped arrays are restored with the `.pkl` crop box |
| Other models | `.npz` / `.npy` / `.nii.gz` / `.mha` / `.nrrd` | multi-channel (C × …) or single foreground map |

* **Logits** (values outside [0, 1]) are detected and converted with softmax (multi-channel) or sigmoid (single channel).
* **(x, y, z) arrays** are transposed automatically when the reversed shape matches the labels.
* **2D data** works the same way (e.g. PNG masks; values {0, 255} are mapped to {0, 1}; HD95 is reported in pixels). nnU-Net 2D natural-image outputs (C × 1 × H × W) are squeezed automatically.

**Labels:** NIfTI/MHA/NRRD/PNG masks with the same case IDs. **Images** (optional) are only used for output geometry; nnU-Net channel suffixes (`_0000`) are handled.

## Configuration reference

All keys have defaults; a minimal config needs `labels`, `calibration.probabilities`, `calibration.labels`, and either `target.probabilities` or `calibration.split`. Relative paths are resolved from the config file's folder. Full templates: `conseg init-config {nnunet_v2_oof | nnunet_v2_heldout | split_pool | generic_2d}`.

| Key | Default | Meaning |
|---|---|---|
| `labels` | all foreground values | `name: [label values]`; several values = composite target |
| `channels` | = label values | `name: [channel indices]` when channels are not label values (region-based models) |
| `input.format` | `auto` | `nnunet_v2`, `nnunet_v1`, `npz`, `npy`, `image` |
| `input.prob_kind` | `auto` | `softmax`, `sigmoid`, `logits_softmax`, `logits_sigmoid` |
| `input.array_order` | `auto` | `zyx` (SimpleITK/nnU-Net) or `xyz` |
| `input.prob_suffix` / `label_suffix` | auto | restrict files, e.g. `.npz`, `.png` |
| `calibration.probabilities` / `labels` | — | folders, files, or globs (lists allowed) |
| `calibration.split` | `null` | fraction of a labeled pool used for calibration; rest = labeled test set |
| `calibration.patient_map` | `null` | CSV `case,patient` so one patient is never in both sets |
| `calibration.threshold` | `auto` | Dice-optimal threshold, or a number |
| `calibration.alpha` | `auto` | number, or per target `{pancreas: 0.0003}` |
| `target.probabilities` / `labels` / `images` | — | cases to check; labels optional (enable evaluation) |
| `flagging.rule` / `value` | `top_fraction` / `0.2` | see below |
| `flagging.flag_undefined` | `true` | flag cases with uncertain but no certain-foreground voxels |
| `outputs.dir` | `conseg_output` | output folder |
| `outputs.write_maps` / `write_baseline` | `true` | write conformal maps / baseline masks |
| `evaluation.hd95` / `empty_hd95` | `true` / `nan` | HD95; value when a mask is empty (`nan` or `diagonal`) |
| `evaluation.bootstrap` / `permutations` | `1000` / `10000` | resamples for CIs / permutation test |
| `evaluation.retention` | `[0.9, 0.5, 0.2]` | fractions retained for the summary table |
| `workers` | `1` | parallel processes |

## Outputs

```
conseg_output/
├── calibration.json        # thresholds, alpha, NCST, coverage and UR percentiles per target (reusable)
├── cases.csv               # one row per case and target: UR, voxel counts, flags (+ Dice, HD95, coverage if labeled)
├── review_list.csv         # one row per case, ranked: flagged?, which targets, max UR, reason
├── maps/
│   ├── CASE_TARGET_conformal.nii.gz   # 0 certain background, 1 uncertain, 2 certain foreground
│   └── CASE_TARGET_baseline.nii.gz    # model mask at the calibrated threshold
├── evaluation/             # only when reference labels are available
│   ├── summary.csv         # AURCC (UR vs random), 95% CIs, P values, pooled vs target coverage
│   ├── retention.csv       # mean Dice/HD95 at 100/90/50/20% retention
│   └── risk_coverage_TARGET.png
├── config_used.yaml, run.log, run_summary.json
```

Conformal maps open as label overlays in ITK-SNAP or 3D Slicer.

## Flagging cases for review

| `flagging.rule` | `value` | Flags |
|---|---|---|
| `top_fraction` | e.g. `0.2` | the 20% most uncertain of the cases with a defined UR |
| `ur_threshold` | e.g. `1.5` | cases with UR above a fixed value |
| `calibration_percentile` | e.g. `90` | cases with UR above the 90th percentile of the calibration cases — a threshold fixed in advance |

A case is flagged if any target is flagged. UR values are comparable within a model–target pair, not across models.

## Command line and Python API

```bash
conseg run        -c config.yaml        # calibrate + predict + evaluate (if labels)
conseg calibrate  -c config.yaml        # -> calibration.json
conseg predict    -c config.yaml [--calibration path/to/calibration.json]
conseg evaluate   -c config.yaml        # re-run evaluation from cases.csv
conseg inspect    file.npz              # what is inside a probability file
conseg init-config TEMPLATE -o config.yaml
# common options: -o OUTPUT_DIR, -w WORKERS, -v
```

```python
from conseg import load_config, run_all, calibrate, predict
cfg = load_config("config.yaml")
calib, cases, review, summary = run_all(cfg)
```

## Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| *No probability files found* | nnU-Net was run without `--save_probabilities` / `--npz`; re-run inference with it. |
| *could not be paired by case ID* | probability and label file names differ; rename or set `input.prob_suffix` / `label_suffix`. |
| *probability shape … does not match label shape* | the model reoriented images (e.g. nibabel RAS); export probabilities in the original image orientation, or set `input.array_order`. |
| *degenerate NCST* | α too large or too small for this target's size; set α explicitly (smaller for small structures). |
| *Single-channel probability maps describe one target* | a single foreground map can only define one target; configure one entry in `labels`. |
| Many cases flagged with *UR undefined* | the model finds uncertain but no certain foreground; check the target label values, `channels`, and α. |
| *uses channels …, but the probability maps have only N channels* | region-based or sigmoid model; set `channels` (see [How it works](#how-it-works)). |
| Out of memory | probabilities are processed one case at a time; reduce `workers`. |

## Limitations

* **Coverage is marginal and approximate.** The guarantee refers to all test voxels pooled together, not to each case or to the foreground alone. Voxels within a case are correlated, so pooled coverage can fall slightly below 1 − α (typically by a few tenths of a percentage point). Check `evaluation/summary.csv`.
* **Exchangeability.** Calibration and new cases must come from the same distribution; recalibrate when the scanner, protocol, or population changes.
* **Binary per target.** Each structure is handled one-vs-rest; interactions between structures are not modeled.

