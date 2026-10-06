import numpy as np
import torch
from torch import nn


def calculate_pos_weight(
    label_array,
    device,
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

    pos_weight = negative / positive

    return torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=device,
    )


def build_classification_loss(
    train_labels,
    device,
):
    pos_weight = calculate_pos_weight(
        train_labels,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    return criterion, pos_weight