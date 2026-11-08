import numpy as np
import pandas as pd
import pytest

from src.metrics.localization import (
    cam_iou,
    energy_inside_box,
    iou_column,
    pointing_game,
    resize_heatmap,
    select_iou_threshold,
)


def box_mask(size=224, rows=slice(0, 112), cols=slice(0, 112)):
    mask = np.zeros((size, size), dtype=np.float32)
    mask[rows, cols] = 1
    return mask


def test_pointing_game_hit_and_miss():
    mask = box_mask()
    hit = np.zeros((7, 7)); hit[1, 1] = 1.0
    miss = np.zeros((7, 7)); miss[6, 6] = 1.0
    assert pointing_game(hit, mask) == 1
    assert pointing_game(miss, mask) == 0


def test_pointing_game_uses_cell_center():
    # ô (3, 3) của grid 7x7 có tâm ở pixel 112 -> ngay ngoài box [0, 112)
    mask = box_mask()
    cam = np.zeros((7, 7)); cam[3, 3] = 1.0
    assert pointing_game(cam, mask) == 0


def test_iou_perfect_disjoint_and_partial():
    mask = box_mask()
    assert cam_iou(mask.copy(), mask, 0.5) == pytest.approx(1.0)
    assert cam_iou(box_mask(rows=slice(112, 224), cols=slice(112, 224)), mask, 0.5) == 0.0
    half = box_mask(rows=slice(0, 112), cols=slice(0, 56))  # nằm trọn trong GT, bằng 1/2 diện tích
    assert cam_iou(half, mask, 0.5) == pytest.approx(0.5, abs=1e-6)


def test_iou_threshold_effect():
    mask = box_mask()
    heat = mask * 0.4
    assert cam_iou(heat, mask, 0.3) == pytest.approx(1.0)
    assert cam_iou(heat, mask, 0.5) == 0.0


def test_eib_inside_outside_and_mixed():
    mask = box_mask()
    assert energy_inside_box(mask.copy(), mask) == pytest.approx(1.0, abs=1e-6)
    assert energy_inside_box(1 - mask, mask) == pytest.approx(0.0, abs=1e-6)
    # đều khắp ảnh -> EIB = tỉ lệ diện tích box (1/4)
    assert energy_inside_box(np.ones_like(mask), mask) == pytest.approx(0.25, abs=1e-6)


def test_shape_mismatch_raises():
    mask = box_mask()
    with pytest.raises(ValueError):
        cam_iou(np.zeros((7, 7)), mask, 0.5)
    with pytest.raises(ValueError):
        energy_inside_box(np.zeros((7, 7)), mask)


def test_resize_heatmap_shape_and_range():
    grid = np.random.default_rng(0).random((14, 14)).astype(np.float32)
    out = resize_heatmap(grid, 224)
    assert out.shape == (224, 224)
    assert out.min() >= grid.min() - 1e-6 and out.max() <= grid.max() + 1e-6


def test_select_iou_threshold_picks_best_and_breaks_ties_low():
    thresholds = [0.2, 0.4, 0.6]
    records = pd.DataFrame({
        iou_column(0.2): [0.1, 0.2],
        iou_column(0.4): [0.5, 0.7],
        iou_column(0.6): [0.3, 0.3],
    })
    best, means = select_iou_threshold(records, thresholds)
    assert best == 0.4
    assert means[0.4] == pytest.approx(0.6)

    tied = pd.DataFrame({iou_column(t): [0.5] for t in thresholds})
    assert select_iou_threshold(tied, thresholds)[0] == 0.2


def test_select_iou_threshold_rejects_empty():
    with pytest.raises(ValueError):
        select_iou_threshold(pd.DataFrame(columns=[iou_column(0.5)]), [0.5])
