import numpy as np
import pytest

from scripts.evaluate import (
    build_predictions_table,
    select_decision_thresholds,
)
from src.metrics.classification import THRESHOLD_STRATEGY
from src.utils.checksums import file_sha256
from src.data.protocol import validate_checkpoint_protocol


def test_checkpoint_protocol_accepts_matching_split_and_rejects_old_split():
    checkpoint = {
        "model_name": "resnet50",
        "findings": ["a", "b"],
        "split_assignment_sha256": "new-split",
        "data_protocol_id": "protocol-v1",
    }
    validate_checkpoint_protocol(
        checkpoint,
        model_name="resnet50",
        findings=["a", "b"],
        assignment_sha256="new-split",
        protocol_id="protocol-v1",
    )

    with pytest.raises(ValueError, match="split fingerprint"):
        validate_checkpoint_protocol(
            checkpoint,
            model_name="resnet50",
            findings=["a", "b"],
            assignment_sha256="old-split",
            protocol_id="protocol-v1",
        )

    with pytest.raises(ValueError, match="data protocol"):
        validate_checkpoint_protocol(
            checkpoint,
            model_name="resnet50",
            findings=["a", "b"],
            assignment_sha256="new-split",
            protocol_id="protocol-v2",
        )


def test_predictions_table_has_one_row_per_image_and_all_outputs():
    table = build_predictions_table(
        image_ids=["image_a", "image_b", "image_c"],
        targets=np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.uint8),
        probabilities=np.asarray(
            [[0.9, 0.1], [0.4, 0.6], [0.5, 0.49]],
            dtype=float,
        ),
        findings=["finding_a", "finding_b"],
        threshold=0.5,
    )

    assert len(table) == 3
    assert list(table["image_id"]) == ["image_a", "image_b", "image_c"]
    assert set(table.columns) == {
        "image_id",
        "target::finding_a",
        "probability::finding_a",
        "prediction::finding_a",
        "target::finding_b",
        "probability::finding_b",
        "prediction::finding_b",
    }
    assert list(table["prediction::finding_a"]) == [1, 0, 1]
    assert list(table["prediction::finding_b"]) == [0, 1, 0]


def test_predictions_table_supports_per_class_thresholds():
    table = build_predictions_table(
        image_ids=["image_a", "image_b"],
        targets=np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
        probabilities=np.asarray([[0.4, 0.4], [0.3, 0.6]], dtype=float),
        findings=["finding_a", "finding_b"],
        threshold={"finding_a": 0.35, "finding_b": 0.5},
    )

    assert list(table["prediction::finding_a"]) == [1, 0]
    assert list(table["prediction::finding_b"]) == [0, 1]


def test_evaluation_uses_checkpoint_thresholds_and_allows_scalar_override(tmp_path):
    validation_manifest = tmp_path / "val_manifest.csv"
    validation_manifest.write_text("image_id\nexample\n", encoding="utf-8")
    checkpoint = {
        "decision_thresholds": {"a": 0.25, "b": 0.75},
        "threshold_strategy": THRESHOLD_STRATEGY,
        "threshold_validation_manifest_sha256": file_sha256(validation_manifest),
    }

    thresholds, source, strategy = select_decision_thresholds(
        checkpoint,
        findings=["a", "b"],
        threshold_override=None,
        validation_manifest_path=validation_manifest,
    )
    assert thresholds == {"a": 0.25, "b": 0.75}
    assert source == "checkpoint"
    assert strategy == THRESHOLD_STRATEGY

    thresholds, source, strategy = select_decision_thresholds(
        {},
        findings=["a", "b"],
        threshold_override=0.4,
        validation_manifest_path=validation_manifest,
    )
    assert thresholds == {"a": 0.4, "b": 0.4}
    assert source == "cli_override"
    assert strategy == "scalar_override"


def test_evaluation_rejects_checkpoint_without_tuned_thresholds(tmp_path):
    validation_manifest = tmp_path / "val_manifest.csv"
    validation_manifest.write_text("image_id\nexample\n", encoding="utf-8")

    with pytest.raises(ValueError, match="decision_thresholds"):
        select_decision_thresholds(
            {},
            findings=["a"],
            threshold_override=None,
            validation_manifest_path=validation_manifest,
        )
