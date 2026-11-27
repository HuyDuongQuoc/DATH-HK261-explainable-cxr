import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.factory import build_model
from src.models.resnet import forward_with_activation
from src.models.vit_gap import ViTGAP
from src.training.engine import run_epoch
from src.xai.evaluation import collect_localization_records
from src.xai.gradcam import GradCAM


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = build_model("vit_b_16_gap", 14, pretrained=False).eval()
    # the random head init is small; scale it up so the CAM has contrast
    with torch.no_grad():
        m.head.weight.mul_(50)
    return m


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(1)
    return torch.randn(2, 3, 224, 224)


def test_builds_wrapper_without_cls_head(model):
    assert isinstance(model, ViTGAP)
    assert isinstance(model.backbone.heads, torch.nn.Identity)
    assert model.head.out_features == 14


def test_encode_matches_torchvision_encoder(model, images):
    vit = model.backbone
    with torch.no_grad():
        x = vit._process_input(images)
        x = torch.cat([vit.class_token.expand(len(x), -1, -1), x], dim=1)
        reference = vit.encoder(x)
        assert torch.allclose(model.encode(images), reference, atol=1e-5)


def test_cam_is_14x14_and_mean_equals_logits(model, images):
    with torch.no_grad():
        logits, activation = forward_with_activation(model, "vit_b_16_gap", images)
        assert activation.shape == (2, 14, 14, 14)
        assert torch.allclose(activation.mean(dim=(2, 3)), logits, atol=1e-4)
        assert torch.allclose(logits, model(images), atol=1e-4)


def test_gradcam_last_equals_cam_without_bias(model, images):
    with GradCAM(model, "vit_b_16_gap") as gradcam:
        cam, _ = gradcam.generate(images, 3)
    with torch.no_grad():
        _, activation = forward_with_activation(model, "vit_b_16_gap", images)
        ref = F.relu(activation[:, 3] - model.head.bias[3]).numpy()
    assert cam.shape == (2, 14, 14)
    for i in range(len(images)):
        r = (ref[i] - ref[i].min()) / (ref[i].max() - ref[i].min() + 1e-12)
        assert np.allclose(r, cam[i], atol=1e-4)


def test_gradcam_penultimate_runs(model, images):
    with GradCAM(model, "vit_b_16_gap", target_layer="penultimate") as gradcam:
        cam, _ = gradcam.generate(images, 3)
    assert cam.shape == (2, 14, 14)
    assert cam.min() >= 0 and cam.max() <= 1
    assert gradcam.gradients is not None and gradcam.gradients.abs().sum() > 0


def test_localization_loss_trains_vit_gap():
    torch.manual_seed(0)
    m = build_model("vit_b_16_gap", 14, pretrained=False)
    union = torch.zeros(2, 14, 224, 224)
    union[:, :, :112, :112] = 1
    valid = torch.zeros(2, 14, dtype=torch.bool)
    valid[:, 3] = True
    labels = torch.zeros(2, 14)
    labels[:, 3] = 1
    batch = {
        "image": torch.randn(2, 3, 224, 224),
        "labels": labels,
        "image_id": ["a", "b"],
        "union_masks": union,
        "soft_masks": union,
        "mask_valid": valid,
    }
    opt = torch.optim.AdamW(m.parameters(), 1e-4)
    result = run_epoch(
        m, [batch], torch.nn.BCEWithLogitsLoss(), torch.device("cpu"),
        optimizer=opt, model_name="vit_b_16_gap", loc_weight=1.0,
    )
    assert np.isfinite(result["loss"]) and result["loc_loss"] > 0


def test_localization_records_for_vit_gap(model):
    union = torch.zeros(1, 14, 224, 224)
    union[:, :, :112, :112] = 1
    valid = torch.zeros(1, 14, dtype=torch.bool)
    valid[0, 2] = True
    batches = [{
        "image": torch.randn(1, 3, 224, 224),
        "union_masks": union,
        "mask_valid": valid,
        "image_id": ["x"],
    }]
    records = collect_localization_records(
        model, "vit_b_16_gap", batches, [f"f{i}" for i in range(14)],
        torch.device("cpu"), thresholds=[0.5],
    )
    assert len(records) == 1 and 0 <= records["eib"].iloc[0] <= 1


def test_dilation_rejected_for_vit_gap():
    with pytest.raises(ValueError):
        build_model("vit_b_16_gap", 14, pretrained=False, dilate_last_stage=True)
