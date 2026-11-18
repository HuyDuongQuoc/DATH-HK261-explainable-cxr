import pytest
import torch

from src.losses.localization_loss import localization_loss


def make_mask(batch=1, classes=2, size=224):
    mask = torch.zeros(batch, classes, size, size)
    mask[:, :, :112, :112] = 1
    return mask


def peaked(top, left, size=7, high=20.0, low=-20.0, batch=1, classes=2):
    act = torch.full((batch, classes, size, size), low)
    act[:, :, top:top + 3, left:left + 3] = high
    return act


def test_low_inside_high_outside():
    mask = make_mask()
    valid = torch.tensor([[True, True]])
    inside = localization_loss(peaked(0, 0), mask, valid)
    outside = localization_loss(peaked(4, 4), mask, valid)
    assert inside.item() < 0.05
    assert outside.item() > 0.95
    assert inside < outside


def test_only_valid_pairs_are_averaged():
    mask = make_mask()
    # class 0 nằm trong box (loss ~0), class 1 nằm ngoài (loss ~1)
    act = torch.cat([peaked(0, 0, classes=1), peaked(4, 4, classes=1)], dim=1)
    only_first = localization_loss(act, mask, torch.tensor([[True, False]]))
    only_second = localization_loss(act, mask, torch.tensor([[False, True]]))
    both = localization_loss(act, mask, torch.tensor([[True, True]]))
    assert only_first.item() < 0.05
    assert only_second.item() > 0.95
    assert both.item() == pytest.approx((only_first.item() + only_second.item()) / 2, abs=1e-5)


def test_no_valid_pair_returns_zero_but_keeps_graph():
    act = peaked(4, 4).requires_grad_()
    loss = localization_loss(act, make_mask(), torch.zeros(1, 2, dtype=torch.bool))
    assert loss.item() == 0.0
    loss.backward()
    assert act.grad is not None and torch.isfinite(act.grad).all()


def test_gradient_flows_and_is_finite():
    act = peaked(4, 4).requires_grad_()
    loss = localization_loss(act, make_mask(), torch.tensor([[True, True]]))
    loss.backward()
    assert act.grad.abs().sum() > 0
    assert torch.isfinite(act.grad).all()


def test_gradient_pushes_energy_into_box():
    act = torch.zeros(1, 1, 7, 7, requires_grad=True)
    loss = localization_loss(act, make_mask(classes=1), torch.tensor([[True]]))
    loss.backward()
    # giảm loss = tăng activation trong box (grad âm), giảm ngoài box (grad dương)
    assert act.grad[0, 0, 0, 0] < 0
    assert act.grad[0, 0, 6, 6] > 0


def test_scale_invariant():
    mask = make_mask()
    valid = torch.tensor([[True, True]])
    a = localization_loss(peaked(0, 0), mask, valid)
    b = localization_loss(peaked(0, 0) * 3, mask, valid)
    assert a.item() == pytest.approx(b.item(), abs=1e-4)


def test_matching_resolution_skips_interpolation():
    mask = make_mask(size=8)
    act = torch.full((1, 2, 8, 8), -20.0)
    act[:, :, :4, :4] = 20.0
    mask[:, :, :4, :4] = 1
    mask[:, :, 4:, :] = 0
    mask[:, :, :, 4:] = 0
    loss = localization_loss(act, mask, torch.tensor([[True, True]]))
    assert loss.item() < 0.01
