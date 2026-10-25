import numpy as np
import pytest

from scripts.evaluate import build_predictions_table
from src.data.protocol import validate_checkpoint_protocol


def test_checkpoint_protocol_accepts_matching_split_and_rejects_old_split():
    checkpoint = {
        "model_name": "resnet50",
        "findings": ["a", "b"],
        "split_assignment_sha256": "new-split",
    }
    validate_checkpoint_protocol(
        checkpoint,
        model_name="resnet50",
        findings=["a", "b"],
        assignment_sha256="new-split",
    )

    with pytest.raises(ValueError, match="split fingerprint"):
        validate_checkpoint_protocol(
            checkpoint,
            model_name="resnet50",
            findings=["a", "b"],
            assignment_sha256="old-split",
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
