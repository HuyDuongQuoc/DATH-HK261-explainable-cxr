"""
Bootstrap confidence intervals and paired comparisons between two runs.

Image-level (cluster) bootstrap: each replicate resamples image_ids with
replacement, keeps every (image, finding) pair of a sampled image, and computes the
macro average over findings as in the reports. Two runs are compared on the same
resampled images (paired), which gives tighter intervals for the difference than
comparing two separate CIs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _macro_by_finding(frame, metric):
    return frame.groupby("finding")[metric].mean().mean()


def _resample_index(image_ids, rng):
    unique = np.unique(image_ids)
    drawn = rng.choice(unique, size=len(unique), replace=True)
    counts = pd.Series(drawn).value_counts()
    return counts


def _weighted_macro(frame, metric, weights_by_image):
    w = frame["image_id"].map(weights_by_image).fillna(0).to_numpy()
    keep = w > 0
    if not keep.any():
        return np.nan
    sub = frame.loc[keep, ["finding", metric]].copy()
    sub["w"] = w[keep]
    sub["wv"] = sub[metric] * sub["w"]
    grouped = sub.groupby("finding")[["wv", "w"]].sum()
    return float((grouped["wv"] / grouped["w"]).mean())


def bootstrap_macro_ci(records, metric, n_boot=1000, alpha=0.05, seed=0):
    """Macro average of `metric` over findings, with an image-level bootstrap CI."""
    rng = np.random.default_rng(seed)
    frame = records.dropna(subset=[metric])
    point = float(_macro_by_finding(frame, metric))
    stats = []
    for _ in range(n_boot):
        weights = _resample_index(frame["image_id"].to_numpy(), rng)
        stats.append(_weighted_macro(frame, metric, weights))
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def paired_localization_diff(records_a, records_b, metric, n_boot=1000, alpha=0.05, seed=0):
    """
    Difference macro(metric) of B minus A on the (image_id, finding) pairs present in both.

    Returns:
        dict with the estimate, the CI and a two-sided p-value (share of bootstrap
        replicates that cross zero).
    """
    keys = ["image_id", "finding"]
    merged = records_a[keys + [metric]].merge(
        records_b[keys + [metric]], on=keys, suffixes=("_a", "_b")
    ).dropna()
    if merged.empty:
        raise ValueError("No shared (image_id, finding) pairs between runs")
    merged["diff"] = merged[f"{metric}_b"] - merged[f"{metric}_a"]

    rng = np.random.default_rng(seed)
    point = float(_macro_by_finding(merged, "diff"))
    stats = []
    for _ in range(n_boot):
        weights = _resample_index(merged["image_id"].to_numpy(), rng)
        stats.append(_weighted_macro(merged, "diff", weights))
    stats = np.asarray(stats)
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return {
        "metric": metric,
        "a": float(_macro_by_finding(merged, f"{metric}_a")),
        "b": float(_macro_by_finding(merged, f"{metric}_b")),
        "diff": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_two_sided": float(min(p, 1.0)),
        "n_pairs": int(len(merged)),
        "n_boot": n_boot,
    }


def _macro_auroc(targets, probabilities):
    scores = []
    for c in range(targets.shape[1]):
        t = targets[:, c]
        if t.min() == t.max():
            continue
        scores.append(roc_auc_score(t, probabilities[:, c]))
    return float(np.mean(scores)) if scores else np.nan


def load_predictions(path, findings):
    """Read test_predictions.csv (columns target::<finding>, probability::<finding>)."""
    frame = pd.read_csv(path).set_index("image_id")
    targets = frame[[f"target::{f}" for f in findings]].to_numpy(dtype=float)
    probabilities = frame[[f"probability::{f}" for f in findings]].to_numpy(dtype=float)
    return frame.index.to_numpy(), targets, probabilities


def paired_auroc_diff(pred_a, pred_b, n_boot=1000, alpha=0.05, seed=0):
    """
    Macro-AUROC difference (B minus A) with a paired image-level bootstrap.

    pred_a and pred_b are (image_ids, targets, probabilities) from load_predictions.
    """
    ids_a, t_a, p_a = pred_a
    ids_b, t_b, p_b = pred_b
    order_b = pd.Series(np.arange(len(ids_b)), index=ids_b).reindex(ids_a)
    if order_b.isna().any():
        raise ValueError("Prediction files do not cover the same images")
    idx_b = order_b.to_numpy(dtype=int)
    t_b, p_b = t_b[idx_b], p_b[idx_b]
    if not np.array_equal(t_a, t_b):
        raise ValueError("Targets differ between prediction files")

    rng = np.random.default_rng(seed)
    point = _macro_auroc(t_a, p_b) - _macro_auroc(t_a, p_a)
    stats = []
    n = len(ids_a)
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        stats.append(_macro_auroc(t_a[s], p_b[s]) - _macro_auroc(t_a[s], p_a[s]))
    stats = np.asarray(stats)
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return {
        "metric": "macro_auroc",
        "a": _macro_auroc(t_a, p_a),
        "b": _macro_auroc(t_a, p_b),
        "diff": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_two_sided": float(min(p, 1.0)),
        "n_images": int(n),
        "n_boot": n_boot,
    }
