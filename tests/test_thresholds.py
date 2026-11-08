import numpy as np
import pytest

from src.metrics.classification import (
    THRESHOLD_STRATEGY,
    compute_classification_metrics,
    optimize_f1_thresholds,
    resolve_decision_thresholds,
)


def test_per_class_thresholds_are_deterministic_and_do_not_reduce_f1():
    findings = ["finding_a", "finding_b"]
    targets = np.asarray(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    probabilities = np.asarray(
        [
            [0.45, 0.20],
            [0.40, 0.70],
            [0.80, 0.35],
            [0.30, 0.30],
            [0.55, 0.10],
            [0.52, 0.60],
        ],
        dtype=float,
    )

    first = optimize_f1_thresholds(targets, probabilities, findings)
    second = optimize_f1_thresholds(targets, probabilities, findings)
    tuned = compute_classification_metrics(
        targets,
        probabilities,
        findings,
        threshold=first,
    )
    fixed = compute_classification_metrics(
        targets,
        probabilities,
        findings,
        threshold=0.5,
    )

    assert first == second
    assert list(first) == findings
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in first.values())
    for tuned_row, fixed_row in zip(tuned["per_class"], fixed["per_class"]):
        assert tuned_row["f1"] >= fixed_row["f1"]


def test_threshold_tie_chooses_closest_to_half_then_larger():
    targets = np.asarray([[1], [0], [0], [1]], dtype=np.uint8)
    probabilities = np.asarray([[0.60], [0.55], [0.45], [0.40]], dtype=float)

    optimized = optimize_f1_thresholds(targets, probabilities, ["finding"])

    assert optimized == {"finding": pytest.approx(0.60)}


def test_metrics_accept_scalar_vector_and_mapping_thresholds():
    targets = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
    probabilities = np.asarray([[0.6, 0.4], [0.4, 0.6]], dtype=float)
    findings = ["a", "b"]

    scalar = compute_classification_metrics(targets, probabilities, findings, 0.5)
    vector = compute_classification_metrics(targets, probabilities, findings, [0.5, 0.5])
    mapping = compute_classification_metrics(
        targets,
        probabilities,
        findings,
        {"a": 0.5, "b": 0.5},
    )

    assert scalar == vector == mapping
    assert resolve_decision_thresholds(0.5, findings).tolist() == [0.5, 0.5]
    with pytest.raises(ValueError, match="finding count"):
        resolve_decision_thresholds([0.5], findings)
    with pytest.raises(ValueError, match="finding names"):
        resolve_decision_thresholds({"a": 0.5}, findings)


def test_threshold_strategy_is_stable_protocol_value():
    assert THRESHOLD_STRATEGY == "per_class_max_f1"
