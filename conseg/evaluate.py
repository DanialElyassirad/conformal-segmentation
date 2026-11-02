"""Evaluation on labeled target cases: coverage, Dice/HD95, risk-coverage analysis and plots."""
import logging
import os

import numpy as np
import pandas as pd

from . import metrics

log = logging.getLogger("conseg")


def evaluate(cfg, calib, df, outdir):
    if "dice_baseline" not in df or df["dice_baseline"].isna().all():
        log.info("No reference labels for the target cases; evaluation skipped")
        return None
    ev = cfg["evaluation"]
    os.makedirs(outdir, exist_ok=True)
    summary, retention, curves = [], [], {}
    for name, d in df[df["dice_baseline"].notna()].groupby("target", sort=False):
        t = calib["targets"][name]
        cov = float((d["coverage"] * d["n_voxels"]).sum() / d["n_voxels"].sum())
        risks = {"DSC": 1 - d["dice_baseline"].to_numpy(float)}
        units = d["hd95_units"].iloc[0] if "hd95_units" in d else ""
        if ev["hd95"] and "hd95_baseline" in d:
            risks[f"HD95 ({units})"] = d["hd95_baseline"].to_numpy(float)
        ur = d["UR"].to_numpy(float)
        for metric, r in risks.items():
            rc = metrics.risk_coverage_analysis(ur, r, ev["bootstrap"], ev["permutations"], ev["seed"])
            is_dsc = metric == "DSC"
            conv = (lambda x: 1 - x) if is_dsc else (lambda x: x)
            row = {"target": name, "metric": metric, "n_cases": int(len(d)), "n_with_UR": rc.get("n", 0),
                   "pooled_coverage": cov, "target_coverage": t["target_coverage"],
                   "mean_all": float(conv(np.nanmean(r)))}
            if rc.get("n", 0) >= 3:
                row.update({"AURCC_UR": rc["aurcc"], "AURCC_UR_CI_low": rc["aurcc_ci"][0],
                            "AURCC_UR_CI_high": rc["aurcc_ci"][1], "AURCC_random": rc["aurcc_random"],
                            "delta_AURCC": rc["delta"], "delta_CI_low": rc["delta_ci"][0],
                            "delta_CI_high": rc["delta_ci"][1], "P_two_sided": rc["p_two_sided"]})
                rm = metrics.retention_means(ur, r, [1.0] + list(ev["retention"]))
                retention.append({"target": name, "metric": metric,
                                  **{f"{int(k * 100)}%": float(conv(v)) for k, v in rm.items()}})
                curves[(name, metric)] = (metrics.rc_curve(ur, r), rc)
            summary.append(row)
    summ = pd.DataFrame(summary)
    summ.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    pd.DataFrame(retention).to_csv(os.path.join(outdir, "retention.csv"), index=False)
    _plots(curves, outdir)
    for _, row in summ.iterrows():
        if "AURCC_UR" in row and pd.notna(row.get("AURCC_UR")):
            log.info("[%s | %s] AURCC %.4g vs random %.4g (%s); coverage %.4f (target %.4f)",
                     row["target"], row["metric"], row["AURCC_UR"], row["AURCC_random"],
                     metrics.format_p(row["P_two_sided"]), row["pooled_coverage"], row["target_coverage"])
    return summ


def _plots(curves, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:   # pragma: no cover
        log.warning("matplotlib not available; plots skipped")
        return
    by_target = {}
    for (name, metric), v in curves.items():
        by_target.setdefault(name, []).append((metric, v))
    for name, items in by_target.items():
        fig, axes = plt.subplots(1, len(items), figsize=(3.6 * len(items), 3.4), squeeze=False)
        for ax, (metric, ((cov, mrisk), rc)) in zip(axes[0], items):
            is_dsc = metric == "DSC"
            y = 1 - mrisk if is_dsc else mrisk
            yr = 1 - rc["aurcc_random"] if is_dsc else rc["aurcc_random"]
            ax.plot(cov, y, lw=1.8, color="#1f5fa8",
                    label=f"UR ranking (AURCC = {rc['aurcc']:.3g}; {metrics.format_p(rc['p_two_sided'])})")
            ax.axhline(yr, ls=":", lw=1.2, color="0.3", label=f"Random (AURCC = {rc['aurcc_random']:.3g})")
            ax.set_xlim(1, 0)
            ax.set_xlabel("Coverage (fraction of cases retained)")
            ax.set_ylabel(f"Mean {'DSC' if is_dsc else metric}\nof retained cases")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=7)
        fig.suptitle(name, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"risk_coverage_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
