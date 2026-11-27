"""
RQ1: does better classification come with better localization?

Three views, all on the run x finding table:

1. Architecture level: macro classification vs macro localization of the baselines
   (λ=0), mean ± sd over seeds, with ranks.
2. Within finding: for each finding, the Spearman correlation between a
   classification and a localization metric across baseline runs (architectures and
   seeds), averaged over findings. This removes the "easy findings score high on
   both" effect. The CI is a bootstrap over findings. The pooled Spearman (all
   run x finding pairs) is also reported but is dominated by finding difficulty.
3. Intervention: compare each λ>0 run with the baseline of the same architecture and
   seed. If localization changes a lot while classification barely moves, the two
   axes are decoupled.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.rq_data import markdown_table

CLS_METRICS = ("auroc", "auprc")
LOC_METRICS = ("pointing_hit_rate", "eib", "eib_over_area", "eib_minus_prior", "iou_mean")


def _available(table, metrics):
    return [m for m in metrics if m in table.columns and table[m].notna().any()]


def architecture_table(table):
    base = table[table["lambda"] == 0]
    metrics = _available(base, CLS_METRICS + LOC_METRICS)
    per_run = base.groupby(["arch", "run"])[metrics].mean()  # macro theo finding
    summary = per_run.groupby("arch").agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "n_seeds", per_run.groupby("arch").size())
    for metric in metrics:
        summary[f"{metric}_rank"] = summary[f"{metric}_mean"].rank(ascending=False, method="min")
    return summary


def _spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
        return np.nan
    return float(spearmanr(x[mask], y[mask]).statistic)


def correlation_table(table, n_boot=2000, seed=0, baseline_only=True):
    data = table[table["lambda"] == 0] if baseline_only else table
    rng = np.random.default_rng(seed)
    findings = data["finding"].unique()
    rows = []
    for cls_metric in _available(data, CLS_METRICS):
        for loc_metric in _available(data, LOC_METRICS):
            per_finding = {}
            for finding, group in data.groupby("finding"):
                per_finding[finding] = _spearman(
                    group[cls_metric].to_numpy(float), group[loc_metric].to_numpy(float)
                )
            values = np.array([per_finding[f] for f in findings], dtype=float)
            within = np.nanmean(values) if np.isfinite(values).any() else np.nan
            boots = []
            for _ in range(n_boot):
                sample = rng.choice(values, size=len(values), replace=True)
                if np.isfinite(sample).any():
                    boots.append(np.nanmean(sample))
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
            pooled = _spearman(data[cls_metric].to_numpy(float), data[loc_metric].to_numpy(float))
            rows.append({
                "cls_metric": cls_metric,
                "loc_metric": loc_metric,
                "within_finding_spearman": within,
                "within_ci_low": lo,
                "within_ci_high": hi,
                "n_findings": int(np.isfinite(values).sum()),
                "runs_per_finding": int(data.groupby("finding")["run"].nunique().median()),
                "pooled_spearman": pooled,
                "n_units": int(len(data)),
            })
    return pd.DataFrame(rows)


def intervention_table(table):
    """Difference (λ>0 run minus the same-arch, same-seed baseline) of each macro metric."""
    metrics = _available(table, CLS_METRICS + LOC_METRICS)
    macro = table.groupby(["arch", "seed", "lambda", "run"])[metrics].mean().reset_index()
    base = macro[macro["lambda"] == 0].set_index(["arch", "seed"])
    rows = []
    for row in macro[macro["lambda"] > 0].to_dict("records"):
        key = (row["arch"], row["seed"])
        if key not in base.index:
            continue
        ref = base.loc[key]
        if isinstance(ref, pd.DataFrame):
            ref = ref.iloc[0]
        out = {"arch": row["arch"], "seed": row["seed"], "lambda": row["lambda"],
               "run": row["run"], "baseline_run": ref["run"]}
        for metric in metrics:
            out[f"delta_{metric}"] = row[metric] - ref[metric]
        rows.append(out)
    return pd.DataFrame(rows)


def to_markdown(arch, corr, intervention, split, method):
    lines = [f"# RQ1 — classification vs localization ({split}, {method})", ""]
    lines += ["## By architecture (baselines λ=0, macro over findings)", "", markdown_table(arch), ""]
    notes = []
    if {"eib_mean", "eib_minus_prior_mean"} <= set(arch.columns):
        prior_eib = (arch["eib_mean"] - arch["eib_minus_prior_mean"]).mean()
        notes.append(
            f"- The anatomical prior (mean training union mask, image-independent) has EIB ≈ {prior_eib:.3f}. "
            "`eib_minus_prior` < 0 means the explanation is less concentrated in the box than this "
            "fixed location map."
        )
    if method == "gradcam":
        notes.append(
            "- Grad-CAM pointing uses each model's native grid (CNN 7×7, ViT 14×14), so it is not "
            "fully comparable between CNN and ViT; occlusion uses the same 7×7 grid for every model."
        )
    if notes:
        lines += ["Notes:", "", *notes, ""]
    lines += [
        "## Spearman correlation",
        "",
        "`within_finding_spearman`: Spearman across baseline runs within each finding, "
        "averaged over findings (bootstrap CI over findings). `pooled_spearman` pools all "
        "run × finding pairs and is therefore dominated by finding difficulty.",
        "",
        markdown_table(corr, index=False, digits=3),
        "",
    ]
    if not intervention.empty:
        lines += [
            "## Intervention with L_loc (λ>0 minus the same-architecture, same-seed baseline)",
            "",
            markdown_table(intervention, index=False),
            "",
        ]
    return "\n".join(lines)
