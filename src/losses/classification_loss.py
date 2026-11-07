import numpy as np
import torch
from torch import nn


LEGACY_RAW_LOSS_PROTOCOL = {
    "name": "weighted_bce",
    "positive_weight": {"mode": "raw_ratio", "max_value": None},
}


def calculate_pos_weight(
    label_array,
    device,
    positive_weight_config=None,
):
    label_array = np.asarray(
        label_array,
        dtype=np.float32,
    )

    positive = label_array.sum(axis=0)
    negative = len(label_array) - positive

    if (positive == 0).any():
        raise ValueError("A training class has no positives")

    if (negative == 0).any():
        raise ValueError("A training class has no negatives")

    raw_pos_weight = negative / positive
    positive_weight_config = positive_weight_config or dict(
        LEGACY_RAW_LOSS_PROTOCOL["positive_weight"]
    )
    mode = positive_weight_config["mode"]
    if mode == "raw_ratio":
        effective_pos_weight = raw_pos_weight
    elif mode == "sqrt_ratio":
        effective_pos_weight = np.sqrt(raw_pos_weight)
    elif mode == "sqrt_ratio_capped":
        effective_pos_weight = np.minimum(
            np.sqrt(raw_pos_weight),
            float(positive_weight_config["max_value"]),
        )
    else:
        raise ValueError(f"Unsupported positive weight mode: {mode}")

    effective_tensor = torch.tensor(
        effective_pos_weight,
        dtype=torch.float32,
        device=device,
    )
    metadata = {
        "positive_count": positive.astype(int).tolist(),
        "negative_count": negative.astype(int).tolist(),
        "raw_pos_weight": raw_pos_weight.astype(float).tolist(),
        "effective_pos_weight": effective_pos_weight.astype(float).tolist(),
    }
    return effective_tensor, metadata


def build_classification_loss(
    train_labels,
    device,
    loss_config=None,
):
    loss_config = loss_config or LEGACY_RAW_LOSS_PROTOCOL
    if loss_config.get("name") != "weighted_bce":
        raise ValueError(f"Unsupported classification loss: {loss_config.get('name')}")
    pos_weight, metadata = calculate_pos_weight(
        train_labels,
        device=device,
        positive_weight_config=loss_config["positive_weight"],
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    loss_protocol = {
        "name": "weighted_bce",
        "positive_weight": {
            "mode": loss_config["positive_weight"]["mode"],
            "max_value": loss_config["positive_weight"].get("max_value"),
        },
    }
    return criterion, pos_weight, loss_protocol, metadata


def build_loss_from_checkpoint(checkpoint, device):
    if "pos_weight" not in checkpoint:
        raise ValueError("Checkpoint is missing pos_weight")
    loss_protocol = resolve_checkpoint_loss_protocol(checkpoint)
    if loss_protocol.get("name") != "weighted_bce":
        raise ValueError("Unsupported checkpoint loss protocol")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=checkpoint["pos_weight"].to(device)
    )
    return criterion, loss_protocol


def resolve_checkpoint_loss_protocol(checkpoint):
    return checkpoint.get("loss_protocol") or LEGACY_RAW_LOSS_PROTOCOL


def validate_checkpoint_loss_protocol(checkpoint, expected_protocol):
    actual_protocol = resolve_checkpoint_loss_protocol(checkpoint)
    if actual_protocol != expected_protocol:
        raise ValueError("Checkpoint loss protocol mismatch")
    return actual_protocol
