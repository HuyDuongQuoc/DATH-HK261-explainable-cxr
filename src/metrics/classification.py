from collections.abc import Mapping

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


THRESHOLD_STRATEGY = "per_class_max_f1"


def _validate_targets_probabilities(targets, probabilities, findings):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=float)

    if targets.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("Targets and probabilities must be two-dimensional")
    if targets.shape != probabilities.shape:
        raise ValueError("Target/probability shape mismatch")
    if targets.shape[1] != len(findings):
        raise ValueError("Class count mismatch")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Targets must be binary")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be in [0, 1]")

    return targets.astype(np.uint8), probabilities


def resolve_decision_thresholds(threshold, findings):
    findings = list(findings)
    if isinstance(threshold, Mapping):
        if set(threshold) != set(findings):
            raise ValueError("Threshold mapping does not match finding names")
        values = np.asarray([threshold[finding] for finding in findings], dtype=float)
    elif np.isscalar(threshold):
        values = np.full(len(findings), float(threshold), dtype=float)
    else:
        values = np.asarray(threshold, dtype=float)
        if values.ndim != 1 or len(values) != len(findings):
            raise ValueError("Threshold vector must match finding count")

    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Thresholds must be finite and in [0, 1]")
    return values


def threshold_mapping(threshold, findings):
    values = resolve_decision_thresholds(threshold, findings)
    return {
        finding: float(values[index])
        for index, finding in enumerate(findings)
    }


def optimize_f1_thresholds(targets, probabilities, findings):
    targets, probabilities = _validate_targets_probabilities(
        targets,
        probabilities,
        findings,
    )
    optimized = {}

    for class_index, finding in enumerate(findings):
        target = targets[:, class_index]
        probability = probabilities[:, class_index]
        if np.unique(target).size != 2:
            raise ValueError(
                f"F1 threshold optimization requires both classes for {finding}"
            )

        precision, recall, candidates = precision_recall_curve(target, probability)
        if candidates.size == 0:
            raise ValueError(f"No threshold candidates for {finding}")
        precision = precision[:-1]
        recall = recall[:-1]
        denominator = precision + recall
        f1 = np.divide(
            2.0 * precision * recall,
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator > 0,
        )

        best_f1 = float(np.max(f1))
        best = np.flatnonzero(np.isclose(f1, best_f1, rtol=1e-12, atol=1e-12))
        distances = np.abs(candidates[best] - 0.5)
        closest = best[
            np.isclose(distances, distances.min(), rtol=1e-12, atol=1e-12)
        ]
        optimized[finding] = float(np.max(candidates[closest]))

    return optimized


def compute_classification_metrics(
    targets,
    probabilities,
    findings,
    threshold=0.5,
):
    targets, probabilities = _validate_targets_probabilities(
        targets,
        probabilities,
        findings,
    )
    thresholds = resolve_decision_thresholds(threshold, findings)
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.uint8)

    per_class = []

    for class_index, finding in enumerate(findings):
        target = targets[:, class_index]
        probability = probabilities[:, class_index]
        prediction = predictions[:, class_index]

        if np.unique(target).size != 2:
            raise ValueError(
                f"AUROC is undefined for {finding}. "
                "Check the evaluation split."
            )

        precision, recall, f1, _ = precision_recall_fscore_support(
            target,
            prediction,
            average="binary",
            zero_division=0,
        )

        per_class.append({
            "finding": finding,
            "threshold": float(thresholds[class_index]),
            "auroc": float(roc_auc_score(target, probability)),
            "average_precision": float(average_precision_score(target, probability)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "positive_count": int(target.sum()),
            "negative_count": int(len(target) - target.sum()),
        })

    return {
        "per_class": per_class,
        "macro_auroc": float(np.mean([row["auroc"] for row in per_class])),
        "macro_average_precision": float(
            np.mean([row["average_precision"] for row in per_class])
        ),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
    }
