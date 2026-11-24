import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LEGACY_RAW_LOSS_PROTOCOL = {
    "name": "weighted_bce",
    "positive_weight": {"mode": "raw_ratio", "max_value": None},
}


LABEL_POLICIES = ("majority", "ignore_minority", "soft_votes")


def _reduce(loss, weight):
    """Mean trên mọi phần tử; nếu có weight thì trung bình có trọng số (bỏ phần tử weight=0)."""
    if weight is None:
        return loss.mean()
    weight = weight.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


class WeightedBCELoss(nn.Module):
    """Tương đương nn.BCEWithLogitsLoss(pos_weight) nhưng nhận thêm weight theo phần tử."""

    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.detach().clone())

    def forward(self, logits, targets, weight=None):
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        return _reduce(loss, weight)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (Ridnik et al., ICCV 2021) cho multi-label.

    L = -[ y (1-p)^γ+ log p  +  (1-y) p_m^γ- log(1-p_m) ],  p_m = max(p - clip, 0)

    γ- > γ+ giảm mạnh đóng góp của nhãn âm dễ; clip (probability margin) bỏ hẳn
    nhãn âm rất dễ và giảm ảnh hưởng của nhãn âm có thể bị gán sai. pos_weight
    (nếu khác 1) nhân vào phần dương. Tính ở fp32; reduction là mean trên phần tử
    (bài gốc dùng sum), để cùng thang với BCE/focal của dự án.
    """

    def __init__(self, pos_weight, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)
        self.register_buffer("pos_weight", pos_weight.detach().clone())

    def forward(self, logits, targets, weight=None):
        logits = logits.float()
        targets = targets.float()
        p = torch.sigmoid(logits)
        p_neg = p
        if self.clip > 0:
            p_neg = (p - self.clip).clamp(min=0.0)

        log_pos = torch.log(p.clamp(min=self.eps))
        log_neg = torch.log((1.0 - p_neg).clamp(min=self.eps))

        loss_pos = targets * (1.0 - p).pow(self.gamma_pos) * log_pos
        loss_neg = (1.0 - targets) * p_neg.pow(self.gamma_neg) * log_neg
        loss = -(self.pos_weight.to(loss_pos.dtype) * loss_pos + loss_neg)
        return _reduce(loss, weight)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma):
        super().__init__()
        gamma = float(gamma)
        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError("gamma must be finite and non-negative")
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight.detach().clone())

    def forward(self, logits, targets, weight=None):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        probability_t = targets * probabilities + (1.0 - targets) * (
            1.0 - probabilities
        )
        modulation = (1.0 - probability_t).pow(self.gamma)
        return _reduce(modulation * bce, weight)


def build_training_targets(labels, reader_votes, policy, num_readers):
    """
    Target và weight cho classification loss khi TRAIN.

    - majority:        nhãn đa số (2/3 reader), không weight (hành vi cũ).
    - ignore_minority: giữ nhãn đa số nhưng weight=0 cho cặp (ảnh, class) chỉ
                       1 reader đánh dấu (đang bị coi là âm tính) — kiểu U-Ignore
                       của CheXpert.
    - soft_votes:      target = số reader đánh dấu / số reader (0, 1/3, 2/3, 1).

    Validation/test luôn dùng nhãn đa số, không qua hàm này.
    """
    if policy == "majority":
        return labels, None
    if reader_votes is None:
        raise ValueError(f"label_policy={policy} requires reader_votes in the batch")
    votes = reader_votes.to(labels.device, labels.dtype)
    if policy == "ignore_minority":
        minority_negative = (labels == 0) & (votes > 0)
        return labels, (~minority_negative).to(labels.dtype)
    if policy == "soft_votes":
        return (votes / float(num_readers)).clamp(0.0, 1.0), None
    raise ValueError(f"Unknown label policy: {policy}")


def prior_logit_bias(train_labels, pos_weight=None, eps=1e-6):
    """
    Bias khởi tạo cho head: log(p_eff / (1 - p_eff)), với p_eff là tỉ lệ dương
    sau khi tính pos_weight: p_eff = p*w / (p*w + 1 - p). Với raw_ratio, p_eff=0.5
    nên bias=0 (trùng khởi tạo hiện tại); với trọng số nhỏ hơn, bias âm giúp các
    epoch đầu không dự đoán dương tràn lan.
    """
    p = np.asarray(train_labels, dtype=np.float64).mean(axis=0)
    w = np.ones_like(p) if pos_weight is None else np.asarray(pos_weight, dtype=np.float64)
    p_eff = (p * w) / (p * w + (1.0 - p))
    p_eff = np.clip(p_eff, eps, 1.0 - eps)
    return np.log(p_eff / (1.0 - p_eff))


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
    if mode == "none":
        effective_pos_weight = np.ones_like(raw_pos_weight)
    elif mode == "class_balanced":
        # Cui et al. (CVPR 2019): trọng số ∝ 1/effective number (1-β^n)/(1-β).
        # Tỉ lệ trọng số dương/âm của từng class: (1-β^n_neg) / (1-β^n_pos).
        beta = float(positive_weight_config["beta"])
        effective_pos_weight = (1.0 - beta ** negative) / (1.0 - beta ** positive)
    elif mode == "raw_ratio":
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
    loss_name = loss_config.get("name")
    if loss_name not in {"weighted_bce", "sigmoid_focal", "asymmetric"}:
        raise ValueError(f"Unsupported classification loss: {loss_name}")
    pos_weight, metadata = calculate_pos_weight(
        train_labels,
        device=device,
        positive_weight_config=loss_config["positive_weight"],
    )

    loss_protocol = canonical_loss_protocol(loss_config)
    criterion = _criterion_from_protocol(loss_protocol, pos_weight)
    return criterion, pos_weight, loss_protocol, metadata


def canonical_loss_protocol(loss_config):
    """
    Dạng chuẩn của loss config, dùng để lưu vào checkpoint và so khi resume/evaluate.
    Khóa mới chỉ xuất hiện khi khác mặc định, nên protocol của checkpoint cũ không đổi.
    """
    loss_name = loss_config["name"]
    positive_weight = loss_config["positive_weight"]
    protocol = {
        "name": loss_name,
        "positive_weight": {
            "mode": positive_weight["mode"],
            "max_value": positive_weight.get("max_value"),
        },
    }
    if positive_weight["mode"] == "class_balanced":
        protocol["positive_weight"]["beta"] = float(positive_weight["beta"])
    if loss_name == "sigmoid_focal":
        protocol["gamma"] = float(loss_config["gamma"])
    if loss_name == "asymmetric":
        protocol["gamma_pos"] = float(loss_config.get("gamma_pos", 0.0))
        protocol["gamma_neg"] = float(loss_config.get("gamma_neg", 4.0))
        protocol["clip"] = float(loss_config.get("clip", 0.05))
    policy = loss_config.get("label_policy", "majority")
    if policy != "majority":
        protocol["label_policy"] = policy
    return protocol


def _criterion_from_protocol(loss_protocol, pos_weight):
    loss_name = loss_protocol.get("name")
    if loss_name == "weighted_bce":
        return WeightedBCELoss(pos_weight)
    if loss_name == "sigmoid_focal":
        return SigmoidFocalLoss(pos_weight=pos_weight, gamma=loss_protocol["gamma"])
    if loss_name == "asymmetric":
        return AsymmetricLoss(
            pos_weight,
            gamma_pos=loss_protocol["gamma_pos"],
            gamma_neg=loss_protocol["gamma_neg"],
            clip=loss_protocol["clip"],
        )
    raise ValueError("Unsupported loss protocol")


def build_loss_from_checkpoint(checkpoint, device):
    if "pos_weight" not in checkpoint:
        raise ValueError("Checkpoint is missing pos_weight")
    loss_protocol = resolve_checkpoint_loss_protocol(checkpoint)
    loss_name = loss_protocol.get("name")
    pos_weight = checkpoint["pos_weight"].to(device)
    if loss_name not in {"weighted_bce", "sigmoid_focal", "asymmetric"}:
        raise ValueError("Unsupported checkpoint loss protocol")
    criterion = _criterion_from_protocol(loss_protocol, pos_weight)
    return criterion, loss_protocol


def resolve_checkpoint_loss_protocol(checkpoint):
    return checkpoint.get("loss_protocol") or LEGACY_RAW_LOSS_PROTOCOL


def validate_checkpoint_loss_protocol(checkpoint, expected_protocol):
    expected_protocol = canonical_loss_protocol(expected_protocol)
    actual_protocol = resolve_checkpoint_loss_protocol(checkpoint)
    if actual_protocol != expected_protocol:
        raise ValueError("Checkpoint loss protocol mismatch")
    return actual_protocol
