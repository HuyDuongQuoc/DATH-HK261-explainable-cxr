import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.metrics.localization_stats import (
    bootstrap_macro_ci,
    paired_auroc_diff,
    paired_localization_diff,
)
from src.metrics.localization import box_area_fraction, heatmap_correlation
from src.models.factory import build_model
from src.models.resnet import forward_with_activation
from src.xai.evaluation import collect_localization_records, summarize_by_finding
from src.xai.gradcam import GradCAM
from src.xai.occlusion import box_occlusion_drops, occlusion_maps, occlusion_positions
from src.xai.priors import accumulate_class_priors, pool_to_grid

FINDINGS = [f"f{i}" for i in range(14)]


class LeftHalfModel(torch.nn.Module):
    """Class logit = mean of channel 0 over the left half, so only the left half matters."""

    def forward(self, x):
        score = x[:, 0, :, : x.shape[-1] // 2].mean(dim=(1, 2))
        return score[:, None].repeat(1, 14)


def test_occlusion_positions_cover_image():
    assert occlusion_positions(224, 32, 32) == list(range(0, 224, 32))
    assert occlusion_positions(224, 48, 32)[-1] == 224 - 48


def test_occlusion_map_highlights_important_region():
    images = torch.ones(2, 3, 224, 224)
    cams = occlusion_maps(LeftHalfModel(), images, 3, patch=32, stride=32)
    assert cams.shape == (2, 7, 7)
    assert cams[:, :, :3].min() > 0.99      # columns 0-2 lie fully in the left half
    assert cams[:, :, 4:].max() == 0.0


def test_box_occlusion_drops():
    images = torch.ones(1, 3, 224, 224)
    left = torch.zeros(1, 224, 224)
    left[:, :, :112] = 1
    drops = box_occlusion_drops(LeftHalfModel(), images, left, 0)
    assert drops["drop_inside"][0] == pytest.approx(1.0)
    assert drops["drop_outside"][0] == pytest.approx(0.0)


@pytest.mark.parametrize("name", ["resnet50", "densenet121"])
def test_penultimate_gradcam_is_finer_and_differs_from_cam(name):
    torch.manual_seed(0)
    images = torch.randn(2, 3, 224, 224)
    model = build_model(name, 14, pretrained=False).eval()
    with GradCAM(model, name, target_layer="penultimate") as gradcam:
        cam, _ = gradcam.generate(images, 3)
    assert cam.shape == (2, 14, 14)
    assert cam.min() >= 0 and cam.max() <= 1


def test_dilated_resnet_gives_14x14_activation_and_same_logits():
    torch.manual_seed(0)
    images = torch.randn(2, 3, 224, 224)
    model = build_model("resnet50", 14, pretrained=False, dilate_last_stage=True).eval()
    with torch.no_grad():
        logits, activation = forward_with_activation(model, "resnet50", images)
        assert activation.shape == (2, 14, 14, 14)
        assert torch.allclose(logits, model(images), atol=1e-4)
    with pytest.raises(ValueError):
        build_model("densenet121", 14, pretrained=False, dilate_last_stage=True)


def test_activation_is_fp32_under_autocast():
    model = build_model("resnet50", 14, pretrained=False).eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits, activation = forward_with_activation(model, "resnet50", torch.randn(1, 3, 224, 224))
    assert activation.dtype == torch.float32 and logits.dtype == torch.float32


def test_area_and_correlation_metrics():
    mask = np.zeros((10, 10))
    mask[:5, :2] = 1
    assert box_area_fraction(mask) == pytest.approx(0.1)
    a = np.arange(100.0).reshape(10, 10)
    assert heatmap_correlation(a, 2 * a + 1) == pytest.approx(1.0)
    assert np.isnan(heatmap_correlation(np.ones((3, 3)), a[:3, :3]))


def _batches():
    union = torch.zeros(2, 14, 224, 224)
    union[:, :, :112, :112] = 1
    valid = torch.zeros(2, 14, dtype=torch.bool)
    valid[0, 1] = valid[1, 1] = valid[1, 4] = True
    return [{
        "image": torch.randn(2, 3, 224, 224),
        "union_masks": union,
        "mask_valid": valid,
        "image_id": ["img0", "img1"],
    }]


def test_priors_average_valid_masks_only():
    batches = _batches()
    batches[0]["union_masks"][1, 4] = 0
    batches[0]["union_masks"][1, 4, 112:, 112:] = 1
    priors, counts = accumulate_class_priors(batches, 14)
    assert counts[1] == 2 and counts[4] == 1 and counts[0] == 0
    assert priors[1, 0, 0] == 1.0 and priors[4, 0, 0] == 0.0 and priors[4, -1, -1] == 1.0
    assert pool_to_grid(priors[1], (7, 7)).shape == (7, 7)


@pytest.mark.parametrize("method", ["gradcam", "gradcam_penultimate", "occlusion"])
def test_records_with_prior_and_box_occlusion(method):
    torch.manual_seed(0)
    model = build_model("resnet50", 14, pretrained=False)
    batches = _batches()
    priors, _ = accumulate_class_priors(batches, 14)
    records = collect_localization_records(
        model, "resnet50", batches, FINDINGS, torch.device("cpu"),
        thresholds=[0.3, 0.5], method=method, priors=priors, box_occlusion=True,
        occlusion_kwargs={"patch": 112, "stride": 112},
    )
    assert len(records) == 3
    expected = {"box_area", "eib_over_area", "iou_mean", "prior_eib", "prior_hit",
                "eib_minus_prior", "prior_corr", "logit", "drop_inside", "drop_outside"}
    assert expected <= set(records.columns)
    assert np.allclose(records["box_area"], 0.25)
    # the prior equals the mask, so the prior EIB is 1
    assert np.allclose(records["prior_eib"], 1.0)
    table = summarize_by_finding(records, FINDINGS, 0.5)
    assert {"eib_minus_prior", "drop_inside", "iou"} <= set(table.columns)
    assert table.iloc[-1]["finding"] == "MACRO"


def _records(values, finding="f0"):
    return pd.DataFrame({
        "image_id": [f"i{k}" for k in range(len(values))],
        "finding": finding,
        "eib": values,
    })


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    records = _records(rng.uniform(0, 1, 200))
    ci = bootstrap_macro_ci(records, "eib", n_boot=200)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"]


def test_paired_diff_detects_shift_and_null():
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 1, 300)
    a = _records(base)
    shifted = paired_localization_diff(a, _records(base + 0.1), "eib", n_boot=300)
    assert shifted["diff"] == pytest.approx(0.1)
    assert shifted["ci_low"] > 0 and shifted["p_two_sided"] < 0.01
    null = paired_localization_diff(a, _records(base + rng.normal(0, 0.05, 300)), "eib", n_boot=300)
    assert null["ci_low"] < 0 < null["ci_high"]


def test_paired_auroc_diff_identical_runs_is_zero():
    rng = np.random.default_rng(2)
    ids = np.array([f"i{k}" for k in range(200)])
    targets = rng.integers(0, 2, (200, 3)).astype(float)
    probs = rng.uniform(0, 1, (200, 3))
    result = paired_auroc_diff((ids, targets, probs), (ids[::-1], targets[::-1], probs[::-1]), n_boot=100)
    assert result["diff"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0) and result["ci_high"] == pytest.approx(0.0)
