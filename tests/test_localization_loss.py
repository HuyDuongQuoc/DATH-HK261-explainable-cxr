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


# ---------- tùy chọn mới: activation transform và loại mask ----------

from src.losses.localization_loss import select_target_masks


def test_relu_transform_ignores_negative_activation():
    mask = make_mask()
    valid = torch.tensor([[True, True]])
    act = peaked(0, 0, low=-5.0)
    assert localization_loss(act, mask, valid, transform="relu").item() < 1e-6
    # softplus vẫn tính phần âm ngoài box là năng lượng dương nhỏ
    assert localization_loss(act, mask, valid, transform="softplus").item() > 1e-4


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        localization_loss(peaked(0, 0), make_mask(), torch.tensor([[True, True]]), transform="tanh")


def _soft_batch():
    soft = torch.zeros(1, 2, 224, 224)
    soft[:, :, :112, :112] = 1 / 3      # chỉ 1 reader
    soft[:, :, :56, :56] = 2 / 3        # 2 reader
    union = (soft > 0).float()
    return {"soft_masks": soft, "union_masks": union, "mask_valid": torch.tensor([[True, True]])}


def test_select_target_masks_union_majority_soft():
    batch = _soft_batch()
    union, valid_u = select_target_masks(batch, "union")
    majority, valid_m = select_target_masks(batch, "majority")
    soft, _ = select_target_masks(batch, "soft")
    assert union.sum() == 2 * 112 * 112
    assert majority.sum() == 2 * 56 * 56
    assert torch.allclose(soft, batch["soft_masks"])
    assert valid_u.all() and valid_m.all()


def test_majority_drops_pairs_without_agreement():
    batch = _soft_batch()
    batch["soft_masks"][:, 1] = (batch["soft_masks"][:, 1] > 0).float() / 3
    _, valid = select_target_masks(batch, "majority")
    assert valid.tolist() == [[True, False]]


def test_union_mode_matches_previous_behaviour():
    batch = _soft_batch()
    act = torch.randn(1, 2, 7, 7)
    target, valid = select_target_masks(batch, "union")
    old = localization_loss(act, batch["union_masks"], batch["mask_valid"])
    new = localization_loss(act, target, valid)
    assert new.item() == pytest.approx(old.item(), abs=1e-7)


def test_soft_mask_gives_partial_credit():
    batch = _soft_batch()
    # cùng độ phân giải với mask để không có nội suy làm tràn năng lượng
    act = torch.full((1, 2, 224, 224), -20.0)
    act[:, :, :112, :112] = 20.0  # năng lượng phủ vùng 1/3 + 2/3 reader
    soft, valid = select_target_masks(batch, "soft")
    union, _ = select_target_masks(batch, "union")
    assert localization_loss(act, union, valid).item() < 0.01
    assert localization_loss(act, soft, valid).item() > 0.3
