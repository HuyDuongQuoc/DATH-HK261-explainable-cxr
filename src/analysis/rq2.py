"""
RQ2: does the location-aware loss improve localization, and how does classification change?

Each (baseline λ=0, run λ>0) pair with the same architecture and seed is compared
on the external test set with an image-level bootstrap (src/metrics/localization_stats.py):
    - localization with Grad-CAM (the quantity L_loc optimizes) and occlusion (independent)
    - box occlusion: logit drop when hiding the inside/outside of the box (behavioural test)
    - macro AUROC from test_predictions.csv
A per-finding table (EIB and AUROC differences) shows which findings benefit.
"""

import pandas as pd

from src.analysis.rq_data import (
    classification_by_finding,
    markdown_table,
    localization_by_finding,
    localization_records,
    predictions_path,
)
from src.metrics.localization_stats import (
    load_predictions,
    paired_auroc_diff,
    paired_localization_diff,
)

RECORD_METRICS = {
    "gradcam": ("pointing_hit", "eib", "eib_over_area", "eib_minus_prior", "iou_mean"),
    "occlusion": ("pointing_hit", "eib", "eib_over_area", "eib_minus_prior", "iou_mean",
                  "drop_inside", "drop_outside", "drop_inside_norm", "drop_outside_norm"),
}


def add_normalized_drops(records):
    """
    Add drop_*_norm columns: drop_* divided by the std of the run's own logits
    within each finding.

    drop_* are raw logit drops, so they depend on each model's logit scale, which
    L_loc changes. Normalizing keeps paired comparisons free of that scale effect.
    """
    if records is None or "logit" not in records.columns:
        return records
    records = records.copy()
    scale = records.groupby("finding")["logit"].transform("std").replace(0, float("nan"))
    for column in ("drop_inside", "drop_outside"):
        if column in records.columns:
            records[f"{column}_norm"] = records[column] / scale
    return records


def _prediction_findings(path):
    columns = pd.read_csv(path, nrows=0).columns
    return [c.split("::", 1)[1] for c in columns if c.startswith("target::")]


def pairs(runs):
    """Pair each λ>0 run with the baseline of the same model and seed."""
    base = runs[runs["lambda"] == 0].set_index(["model", "seed"])
    out = []
    for row in runs[runs["lambda"] > 0].to_dict("records"):
        key = (row["model"], row["seed"])
        if key in base.index:
            ref = base.loc[key]
            if isinstance(ref, pd.DataFrame):
                ref = ref.iloc[0]
            out.append((ref.to_dict(), row))
    return out


def paired_table(runs, findings=None, n_boot=1000, seed=0, split="test"):
    rows = []
    for base, guided in pairs(runs):
        context = {
            "arch": guided["arch"], "seed": guided["seed"], "lambda": guided["lambda"],
            "baseline": base["run"], "guided": guided["run"],
        }
        pred_a = predictions_path(base["path"], split)
        pred_b = predictions_path(guided["path"], split)
        if pred_a and pred_b:
            names = findings or _prediction_findings(pred_a)
            result = paired_auroc_diff(
                load_predictions(pred_a, names), load_predictions(pred_b, names),
                n_boot=n_boot, seed=seed,
            )
            rows.append(context | {"method": "classification"} | result)
        for method, metrics in RECORD_METRICS.items():
            rec_a = add_normalized_drops(localization_records(base["path"], split, method))
            rec_b = add_normalized_drops(localization_records(guided["path"], split, method))
            if rec_a is None or rec_b is None:
                continue
            for metric in metrics:
                if metric in rec_a.columns and metric in rec_b.columns:
                    result = paired_localization_diff(rec_a, rec_b, metric, n_boot=n_boot, seed=seed)
                    rows.append(context | {"method": method} | result)
    return pd.DataFrame(rows)


def per_finding_table(runs, split="test", method="gradcam"):
    rows = []
    for base, guided in pairs(runs):
        cls_a = classification_by_finding(base["path"], split)
        cls_b = classification_by_finding(guided["path"], split)
        loc_a = localization_by_finding(base["path"], split, method)
        loc_b = localization_by_finding(guided["path"], split, method)
        if any(x is None for x in (cls_a, cls_b, loc_a, loc_b)):
            continue
        a = cls_a.merge(loc_a, on="finding")
        b = cls_b.merge(loc_b, on="finding")
        merged = a.merge(b, on="finding", suffixes=("_base", "_guided"))
        for metric in ("auroc", "auprc", "pointing_hit_rate", "eib", "eib_minus_prior"):
            if f"{metric}_base" in merged and f"{metric}_guided" in merged:
                merged[f"delta_{metric}"] = merged[f"{metric}_guided"] - merged[f"{metric}_base"]
        merged["arch"], merged["seed"], merged["lambda"] = guided["arch"], guided["seed"], guided["lambda"]
        keep = ["arch", "seed", "lambda", "finding", "n_pairs_base"] + [
            c for c in merged.columns if c.startswith("delta_")
        ]
        rows.append(merged[[c for c in keep if c in merged.columns]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def to_markdown(paired, per_finding):
    lines = ["# RQ2 — effect of the location-aware loss (external test)", ""]
    if not paired.empty:
        cols = ["arch", "seed", "lambda", "method", "metric", "a", "b", "diff",
                "ci_low", "ci_high", "p_two_sided"]
        lines += [
            "## Paired comparison (B = λ>0, A = same-architecture, same-seed baseline; image-level bootstrap CI)",
            "",
            "Last-layer Grad-CAM equals the CAM that L_loc optimizes, so localization conclusions "
            "should also rest on occlusion and box occlusion (`drop_inside`, `drop_outside`). "
            "`eib_minus_prior` compares against the image-independent anatomical prior.",
            "",
            "Reading notes:",
            "",
            "- `drop_*` are raw logit drops; L_loc changes the logit scale, so compare with "
            "`drop_*_norm` (divided by the run's own logit std within each finding).",
            "- `drop_outside` hides the whole image outside the box, which is far out of distribution; "
            "an increase does not mean worse localization. The main behavioural evidence is `drop_inside(_norm)`.",
            "- The prior is the same for both runs, so Δ`eib_minus_prior` always equals Δ`eib`; the column "
            "is only informative in its a/b values (negative = worse than the average location map).",
            "",
            markdown_table(paired[[c for c in cols if c in paired.columns]], index=False),
            "",
        ]
    if not per_finding.empty:
        summary = per_finding.groupby("finding")[[c for c in per_finding.columns if c.startswith("delta_")]].mean()
        lines += [
            "## By finding (mean over pairs, Grad-CAM)",
            "",
            markdown_table(
                summary.sort_values("delta_eib", ascending=False)
                if "delta_eib" in summary else summary
            ),
            "",
        ]
    return "\n".join(lines)
