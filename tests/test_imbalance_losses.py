import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.data.reader_votes import build_reader_vote_table
from src.losses.classification_loss import (
    AsymmetricLoss,
    WeightedBCELoss,
    build_classification_loss,
    build_loss_from_checkpoint,
    build_training_targets,
    calculate_pos_weight,
    canonical_loss_protocol,
    prior_logit_bias,
    validate_checkpoint_loss_protocol,
)
from src.training.engine import run_epoch
from src.utils.config import validate_loss_config


def labels_matrix():
    labels = np.zeros((1000, 3), dtype=np.float32)
    labels[:10, 0] = 1     # 1%
    labels[:100, 1] = 1    # 10%
    labels[:500, 2] = 1    # 50%
    return labels


def test_weighted_bce_matches_torch_and_ignores_zero_weight():
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 2, (8, 3)).float()
    pw = torch.tensor([1.0, 2.0, 5.0])
    ours = WeightedBCELoss(pw)(logits, targets)
    ref = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(logits, targets)
    assert ours.item() == pytest.approx(ref.item(), abs=1e-6)

    weight = torch.ones(8, 3)
    weight[:, 0] = 0
    masked = WeightedBCELoss(pw)(logits, targets, weight=weight)
    ref_masked = F.binary_cross_entropy_with_logits(
        logits[:, 1:], targets[:, 1:], pos_weight=pw[1:]
    )
    assert masked.item() == pytest.approx(ref_masked.item(), abs=1e-6)


def test_asl_reduces_to_bce_without_focusing_and_clip():
    torch.manual_seed(0)
    logits = torch.randn(16, 3)
    targets = torch.randint(0, 2, (16, 3)).float()
    asl = AsymmetricLoss(torch.ones(3), gamma_pos=0, gamma_neg=0, clip=0)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    assert asl.item() == pytest.approx(bce.item(), rel=1e-4)


def test_asl_clip_discards_easy_negatives_and_downweights_negatives():
    easy_negative = torch.tensor([[-4.0]])  # p ≈ 0.018 < clip
    zero = torch.zeros(1, 1)
    loss = AsymmetricLoss(torch.ones(1), gamma_neg=4, clip=0.05)
    assert loss(easy_negative, zero).item() == pytest.approx(0.0, abs=1e-7)

    hard_negative = torch.tensor([[1.0]])
    asl_value = loss(hard_negative, zero).item()
    bce_value = F.binary_cross_entropy_with_logits(hard_negative, zero).item()
    assert 0 < asl_value < bce_value


def test_asl_gradient_finite_under_extreme_logits():
    logits = torch.tensor([[-50.0, 50.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    AsymmetricLoss(torch.ones(3))(logits, targets).backward()
    assert torch.isfinite(logits.grad).all()


def test_class_balanced_and_none_pos_weight():
    labels = labels_matrix()
    none, _ = calculate_pos_weight(labels, "cpu", {"mode": "none", "max_value": None})
    assert torch.allclose(none, torch.ones(3))

    cb, meta = calculate_pos_weight(
        labels, "cpu", {"mode": "class_balanced", "max_value": None, "beta": 0.999}
    )
    expected = (1 - 0.999 ** np.array([990, 900, 500])) / (1 - 0.999 ** np.array([10, 100, 500]))
    assert np.allclose(cb.numpy(), expected, rtol=1e-5)
    # rarer class gets more weight, but much less than raw ratio (99)
    assert cb[0] > cb[1] > cb[2] and cb[0] < np.array(meta["raw_pos_weight"])[0]
    assert cb[2] == pytest.approx(1.0, abs=1e-6)


def test_training_targets_policies():
    labels = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    votes = torch.tensor([[1, 2, 0, 3]])
    t, w = build_training_targets(labels, votes, "majority", 3)
    assert w is None and torch.equal(t, labels)
    t, w = build_training_targets(labels, votes, "ignore_minority", 3)
    assert torch.equal(t, labels) and w.tolist() == [[0.0, 1.0, 1.0, 1.0]]
    t, w = build_training_targets(labels, votes, "soft_votes", 3)
    assert w is None and torch.allclose(t, torch.tensor([[1 / 3, 2 / 3, 0.0, 1.0]]))
    with pytest.raises(ValueError):
        build_training_targets(labels, None, "soft_votes", 3)


def test_prior_bias_matches_effective_prevalence():
    labels = labels_matrix()
    bias = prior_logit_bias(labels)
    assert np.allclose(bias, np.log(np.array([0.01, 0.1, 0.5]) / np.array([0.99, 0.9, 0.5])))
    raw = (1 - labels.mean(0)) / labels.mean(0)
    assert np.allclose(prior_logit_bias(labels, raw), 0.0, atol=1e-6)


def _config(**loss):
    base = {"name": "weighted_bce", "positive_weight": {"mode": "raw_ratio", "max_value": None}}
    base.update(loss)
    return {"loss": base}


def test_config_validation_and_canonical_protocol():
    training = _config(label_policy="majority")
    validate_loss_config(training)
    assert canonical_loss_protocol(training["loss"]) == {
        "name": "weighted_bce",
        "positive_weight": {"mode": "raw_ratio", "max_value": None},
    }

    training = {"loss": {"name": "asymmetric", "positive_weight": {"mode": "none", "max_value": None}}}
    validate_loss_config(training)
    assert training["loss"]["gamma_neg"] == 4.0 and training["loss"]["clip"] == 0.05

    for bad in (
        _config(label_policy="u_ones"),
        _config(gamma_neg=4.0),
        {"loss": {"name": "weighted_bce", "positive_weight": {"mode": "class_balanced", "max_value": None, "beta": 1.0}}},
        {"loss": {"name": "weighted_bce", "positive_weight": {"mode": "raw_ratio", "max_value": None, "beta": 0.9}}},
    ):
        with pytest.raises(ValueError):
            validate_loss_config(bad)


def test_checkpoint_round_trip_for_new_losses():
    labels = labels_matrix()
    loss_config = {
        "name": "asymmetric",
        "gamma_pos": 1.0,
        "gamma_neg": 3.0,
        "clip": 0.02,
        "positive_weight": {"mode": "class_balanced", "max_value": None, "beta": 0.99},
        "label_policy": "soft_votes",
    }
    criterion, pos_weight, protocol, _ = build_classification_loss(labels, "cpu", loss_config)
    assert isinstance(criterion, AsymmetricLoss)
    checkpoint = {"pos_weight": pos_weight, "loss_protocol": protocol}
    restored, restored_protocol = build_loss_from_checkpoint(checkpoint, "cpu")
    assert isinstance(restored, AsymmetricLoss) and restored.gamma_neg == 3.0
    validate_checkpoint_loss_protocol(checkpoint, dict(loss_config))
    with pytest.raises(ValueError):
        validate_checkpoint_loss_protocol(checkpoint, {**loss_config, "label_policy": "ignore_minority"})


def test_legacy_checkpoint_still_validates():
    checkpoint = {"pos_weight": torch.ones(3)}
    criterion, _ = build_loss_from_checkpoint(checkpoint, "cpu")
    assert isinstance(criterion, WeightedBCELoss)
    validate_checkpoint_loss_protocol(
        checkpoint,
        {"name": "weighted_bce", "positive_weight": {"mode": "raw_ratio", "max_value": None}},
    )


def test_reader_vote_table(tmp_path):
    csv = tmp_path / "train.csv"
    pd.DataFrame({
        "image_id": ["a", "a", "a", "a", "b", "c"],
        "class_name": ["ILD", "ILD", "ILD", "Nodule/Mass", "No finding", "ILD"],
        "rad_id": ["R1", "R1", "R2", "R3", "R1", "R4"],
    }).to_csv(csv, index=False)
    votes = build_reader_vote_table(csv, ["ILD", "Nodule/Mass"], image_ids=["a", "b", "c"])
    assert votes.loc["a"].tolist() == [2, 1]
    assert votes.loc["b"].tolist() == [0, 0]
    assert votes.loc["c"].tolist() == [1, 0]


def test_engine_applies_policy_only_when_training():
    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x):
            return self.bias.expand(len(x), 2)

    batch = {
        "image": torch.zeros(2, 3, 4, 4),
        "labels": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        "reader_votes": torch.tensor([[1, 2], [0, 0]]),
        "image_id": ["a", "b"],
    }
    criterion = WeightedBCELoss(torch.ones(2))
    model = Constant()
    val = run_epoch(model, [batch], criterion, torch.device("cpu"), label_policy="soft_votes")
    assert val["cls_loss"] == pytest.approx(np.log(2), abs=1e-6)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    train = run_epoch(model, [batch], criterion, torch.device("cpu"), optimizer=opt, label_policy="ignore_minority")
    assert train["cls_loss"] == pytest.approx(np.log(2), abs=1e-6)  # constant logits: mean unaffected
    assert torch.isfinite(torch.tensor(train["loss"]))
