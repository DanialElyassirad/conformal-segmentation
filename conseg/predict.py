"""Apply a calibration to new cases: conformal maps, uncertainty ratio, and the review list."""
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import core, io, metrics
from .calibrate import expect_from
from .data import load_case

log = logging.getLogger("conseg")


def _predict_case(job):
    cid, pp, lp, ip, cfg, calib, outdir = job
    expect = expect_from(calib)
    c = load_case(cid, pp, lp, cfg, expect=expect, image_path=ip, need_geometry=True)
    rows = []
    for name, t in calib["targets"].items():
        p = core.target_probability(c["probs"], c["has_ch"], t.get("channels", t["members"]), c["kind"])
        g = core.target_mask(c["label"], t["members"]) if c["label"] is not None else None
        st, cmap = core.case_statistics(p, t["threshold"], t["ncst"], g)
        row = {"case": cid, "target": name, **st}
        base = (p > t["threshold"])
        if cfg["outputs"]["write_maps"]:
            row["conformal_map"] = io.write_map(cmap, c["geometry"], os.path.join(outdir, "maps", f"{cid}_{name}_conformal"))
        if cfg["outputs"]["write_baseline"]:
            row["baseline_mask"] = io.write_map(base.astype(np.uint8), c["geometry"],
                                                os.path.join(outdir, "maps", f"{cid}_{name}_baseline"))
        if g is not None:
            row["dice_baseline"] = core.dice(base, g)
            row["dice_certain_fg"] = core.dice(cmap == 2, g)
            if cfg["evaluation"]["hd95"]:
                sp = c["spacing"] or (1.0,) * g.ndim
                row["hd95_baseline"] = metrics.hd95(base, g, sp, cfg["evaluation"]["empty_hd95"])
                row["hd95_units"] = c["units"] or "px"
        rows.append(row)
    return rows


def flag_cases(df, calib, cfg):
    """Add rank/flag columns per target and build the per-case review list."""
    rule = cfg["flagging"]["rule"]; val = cfg["flagging"]["value"]; undefined = cfg["flagging"]["flag_undefined"]
    out = []
    for name, d in df.groupby("target", sort=False):
        d = d.copy()
        ur = d["UR"].astype(float)
        # UR undefined (no certain foreground): "unsure" if some voxels are uncertain, "empty" if every
        # voxel is certain background (nothing detected, eg, a structure absent from this case)
        unsure = ur.isna() & (d["n_uncertain"] > 0)
        empty = ur.isna() & (d["n_uncertain"] == 0)
        # rank 1 = most uncertain; unsure cases ranked first when flagged, empty cases last
        key = ur.where(~unsure, np.inf if undefined else -np.inf).fillna(-np.inf)
        d["rank"] = key.rank(ascending=False, method="first").astype(int)
        if rule == "top_fraction":
            # fraction of the cases with a defined UR, so cases without one do not take up review slots
            k = math.ceil(float(val) * int(ur.notna().sum()))
            flag = ur.rank(ascending=False, method="first") <= k
            reason = f"top {float(val):.0%} most uncertain"
        elif rule == "ur_threshold":
            flag = ur > float(val)
            reason = f"UR > {float(val):g}"
        else:
            pct = calib["targets"][name].get("ur_percentiles", {})
            if not pct:
                raise ValueError(f"No calibration UR percentiles for target '{name}'")
            thr = pct[str(int(round(float(val))))]
            flag = ur > thr
            reason = f"UR above calibration percentile {int(round(float(val)))} ({thr:.4g})"
        flag = flag.fillna(False) & ur.notna()
        if undefined:
            flag = flag | unsure
        d["flagged"] = flag
        d["reason"] = np.select([unsure, empty, flag],
                                ["uncertain voxels but no certain-foreground voxel (UR undefined)",
                                 "nothing detected (all voxels certain background)", reason], "")
        out.append(d)
    df = pd.concat(out)
    review = (df.sort_values(["flagged", "UR"], ascending=[False, False], na_position="first")
                .groupby("case", sort=False)
                .agg(flagged=("flagged", "any"),
                     flagged_targets=("target", lambda s: ", ".join(s[df.loc[s.index, "flagged"]])),
                     max_UR=("UR", "max"),
                     reasons=("reason", lambda s: "; ".join(sorted(set(x for x in s if x)))))
                .reset_index()
                .sort_values(["flagged", "max_UR"], ascending=[False, False], na_position="first"))
    return df, review


def predict(cfg, calib, cases, outdir):
    """cases: {case_id: (prob_path, label_path_or_None, image_path_or_None)}"""
    os.makedirs(os.path.join(outdir, "maps"), exist_ok=True)
    cal = {k: v for k, v in calib.items() if not k.startswith("_")}
    jobs = [(c, *cases[c], cfg, cal, outdir) for c in sorted(cases)]
    W = cfg.get("workers", 1)
    log.info("Applying calibration to %d cases", len(jobs))
    if W and W > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=W) as ex:
            rows = [r for rs in ex.map(_predict_case, jobs) for r in rs]
    else:
        rows = [r for j in jobs for r in _predict_case(j)]
    df = pd.DataFrame(rows)
    df, review = flag_cases(df, calib, cfg)
    df.to_csv(os.path.join(outdir, "cases.csv"), index=False)
    review.to_csv(os.path.join(outdir, "review_list.csv"), index=False)
    log.info("%d of %d cases flagged for review -> %s", int(review["flagged"].sum()), len(review),
             os.path.join(outdir, "review_list.csv"))
    return df, review
