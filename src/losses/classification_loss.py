"""
Classification loss used in the paper: weighted BCE with pos_weight = n_neg / n_pos
(raw ratio) per finding, on majority labels (>= 2/3 readers).
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LEGACY_RAW_LOSS_PROTOCOL = {
    "name": "weighted_bce",
    "positive_weight": {"mode": "raw_ratio", "max_value": None},
}


class WeightedBCELoss(nn.Module):
    """Same as nn.BCEWithLogitsLoss(pos_weight), averaged over all elements."""

    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.detach().clone())

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )


def calculate_pos_weight(label_array, device):
    label_array = np.asarray(label_array, dtype=np.float32)
    positive = label_array.sum(axis=0)
    negative = len(label_array) - positive

    if (positive == 0).any():
        raise ValueError("A training class has no positives")
    if (negative == 0).any():
        raise ValueError("A training class has no negatives")

    raw_pos_weight = negative / positive
    effective_tensor = torch.tensor(raw_pos_weight, dtype=torch.float32, device=device)
    metadata = {
        "positive_count": positive.astype(int).tolist(),
        "negative_count": negative.astype(int).tolist(),
        "raw_pos_weight": raw_pos_weight.astype(float).tolist(),
        "effective_pos_weight": raw_pos_weight.astype(float).tolist(),
    }
    return effective_tensor, metadata


def canonical_loss_protocol(loss_config):
    """Canonical loss config, stored in checkpoints and checked on resume/evaluate."""
    if loss_config.get("name") != "weighted_bce":
        raise ValueError(f"Unsupported classification loss: {loss_config.get('name')}")
    positive_weight = loss_config["positive_weight"]
    if positive_weight.get("mode") != "raw_ratio":
        raise ValueError(f"Unsupported positive weight mode: {positive_weight.get('mode')}")
    return {
        "name": "weighted_bce",
        "positive_weight": {"mode": "raw_ratio", "max_value": positive_weight.get("max_value")},
    }


def build_classification_loss(train_labels, device, loss_config=None):
    loss_protocol = canonical_loss_protocol(loss_config or LEGACY_RAW_LOSS_PROTOCOL)
    pos_weight, metadata = calculate_pos_weight(train_labels, device=device)
    return WeightedBCELoss(pos_weight), pos_weight, loss_protocol, metadata


def resolve_checkpoint_loss_protocol(checkpoint):
    return checkpoint.get("loss_protocol") or LEGACY_RAW_LOSS_PROTOCOL


def build_loss_from_checkpoint(checkpoint, device):
    if "pos_weight" not in checkpoint:
        raise ValueError("Checkpoint is missing pos_weight")
    loss_protocol = canonical_loss_protocol(resolve_checkpoint_loss_protocol(checkpoint))
    return WeightedBCELoss(checkpoint["pos_weight"].to(device)), loss_protocol


def validate_checkpoint_loss_protocol(checkpoint, expected_protocol):
    expected_protocol = canonical_loss_protocol(expected_protocol)
    actual_protocol = resolve_checkpoint_loss_protocol(checkpoint)
    if actual_protocol != expected_protocol:
        raise ValueError("Checkpoint loss protocol mismatch")
    return actual_protocol
