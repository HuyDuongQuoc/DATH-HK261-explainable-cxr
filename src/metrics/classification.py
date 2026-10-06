import numpy as np

from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(
    targets,
    probabilities,
    findings,
    threshold=0.5,
):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)

    if targets.shape != probabilities.shape:
        raise ValueError("Target/probability shape mismatch")

    if targets.shape[1] != len(findings):
        raise ValueError("Class count mismatch")

    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    per_class = []

    for class_index, finding in enumerate(findings):
        target = targets[:, class_index]
        probability = probabilities[:, class_index]
        prediction = predictions[:, class_index]

        if np.unique(target).size != 2:
            raise ValueError(
                f"AUROC is undefined for {finding}. "
                "Check the validation split."
            )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                target,
                prediction,
                average="binary",
                zero_division=0,
            )
        )

        per_class.append({
            "finding": finding,
            "auroc": float(
                roc_auc_score(
                    target,
                    probability,
                )
            ),
            "average_precision": float(
                average_precision_score(
                    target,
                    probability,
                )
            ),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "positive_count": int(
                target.sum()
            ),
            "negative_count": int(
                len(target) - target.sum()
            ),
        })

    macro_metrics = {
        "macro_auroc": float(
            np.mean([
                row["auroc"]
                for row in per_class
            ])
        ),
        "macro_average_precision": float(
            np.mean([
                row["average_precision"]
                for row in per_class
            ])
        ),
        "macro_precision": float(
            np.mean([
                row["precision"]
                for row in per_class
            ])
        ),
        "macro_recall": float(
            np.mean([
                row["recall"]
                for row in per_class
            ])
        ),
        "macro_f1": float(
            np.mean([
                row["f1"]
                for row in per_class
            ])
        ),
    }

    return {
        "per_class": per_class,
        **macro_metrics,
    }