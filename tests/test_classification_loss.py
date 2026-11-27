import numpy as np
import pytest
import torch
from torch import nn

from src.losses.classification_loss import (
    LEGACY_RAW_LOSS_PROTOCOL,
    WeightedBCELoss,
    build_classification_loss,
    build_loss_from_checkpoint,
    calculate_pos_weight,
    validate_checkpoint_loss_protocol,
)


def test_weighted_bce_matches_torch():
    torch.manual_seed(0)
    logits, targets = torch.randn(8, 3), torch.randint(0, 2, (8, 3)).float()
    pos_weight = torch.tensor([1.0, 2.0, 5.0])
    expected = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)
    assert torch.allclose(WeightedBCELoss(pos_weight)(logits, targets), expected)


def test_pos_weight_is_raw_ratio():
    labels = np.array([[1, 0], [0, 0], [0, 1], [0, 1]])
    weight, metadata = calculate_pos_weight(labels, "cpu")
    assert torch.allclose(weight, torch.tensor([3.0, 1.0]))
    assert metadata["positive_count"] == [1, 2]


def test_pos_weight_rejects_class_without_positives():
    with pytest.raises(ValueError):
        calculate_pos_weight(np.zeros((4, 2)), "cpu")


def test_checkpoint_round_trip_and_mismatch():
    labels = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    _, pos_weight, protocol, _ = build_classification_loss(labels, "cpu", LEGACY_RAW_LOSS_PROTOCOL)
    checkpoint = {"pos_weight": pos_weight, "loss_protocol": protocol}
    criterion, restored = build_loss_from_checkpoint(checkpoint, "cpu")
    assert isinstance(criterion, WeightedBCELoss) and restored == LEGACY_RAW_LOSS_PROTOCOL
    validate_checkpoint_loss_protocol(checkpoint, LEGACY_RAW_LOSS_PROTOCOL)
    # Old checkpoints without loss_protocol fall back to the raw-ratio protocol.
    validate_checkpoint_loss_protocol({"pos_weight": pos_weight}, LEGACY_RAW_LOSS_PROTOCOL)
    with pytest.raises(ValueError):
        validate_checkpoint_loss_protocol(
            {"loss_protocol": {"name": "sigmoid_focal", "positive_weight": {"mode": "raw_ratio"}}},
            LEGACY_RAW_LOSS_PROTOCOL,
        )
