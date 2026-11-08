from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from scripts.train import (
    build_augmentation_protocol,
    monitor_improved,
    resolve_monitor_score,
    resolve_resume_best_score,
)
from src.data.dataset import CXRDataset
from src.data.transforms import build_image_transform
from src.utils.config import load_config, validate_augmentation_config


AUGMENTATION = {
    "rotation_degrees": 7.0,
    "translate_fraction": 0.03,
    "scale": [0.95, 1.05],
    "intensity_probability": 0.5,
    "brightness": 0.10,
    "contrast": 0.10,
}


def make_gradient_image(size=32):
    values = np.arange(size * size, dtype=np.uint16).reshape(size, size)
    values = (values * 255 // values.max()).astype(np.uint8)
    return Image.fromarray(values, mode="L")


def test_deterministic_transform_is_stable():
    transform = build_image_transform(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    image = make_gradient_image()

    first = transform(image)
    second = transform(image)

    assert torch.equal(first, second)
    assert first.shape == (3, 32, 32)
    assert torch.isfinite(first).all()


def test_augmented_transform_is_seeded_and_changes_image():
    deterministic = build_image_transform(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    augmented = build_image_transform(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        augmentation=AUGMENTATION,
    )
    image = make_gradient_image()

    torch.manual_seed(7)
    first = augmented(image)
    torch.manual_seed(7)
    second = augmented(image)

    assert torch.equal(first, second)
    assert first.shape == (3, 32, 32)
    assert torch.isfinite(first).all()

    baseline = deterministic(image)
    variants = []
    for seed in range(5):
        torch.manual_seed(seed)
        variants.append(augmented(image))
    assert any(not torch.equal(variant, baseline) for variant in variants)


def test_image_only_augmentation_rejects_mask_loading(tmp_path):
    with pytest.raises(ValueError, match="load_masks=True"):
        CXRDataset(
            manifest_path=tmp_path / "unused.csv",
            processed_root=tmp_path,
            findings=["finding"],
            image_size=32,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            load_masks=True,
            augmentation_config=AUGMENTATION,
        )


@pytest.mark.parametrize(
    "config_path,model_name",
    [
        ("configs/resnet50_aug_auprc.yaml", "resnet50"),
        ("configs/densenet121_aug_auprc.yaml", "densenet121"),
        ("configs/vit_aug_auprc.yaml", "vit_b_16"),
    ],
)
def test_aug_auprc_configs_load(config_path, model_name):
    config = load_config(config_path)

    assert config["model"]["name"] == model_name
    assert config["preprocessing"]["use_augmentation"] is True
    assert config["preprocessing"]["augmentation"] == AUGMENTATION
    assert config["training"]["monitor"] == "macro_average_precision"


@pytest.mark.parametrize(
    "field,value",
    [
        ("rotation_degrees", -1.0),
        ("translate_fraction", 0.5),
        ("intensity_probability", 1.1),
        ("brightness", -0.1),
        ("contrast", 1.1),
    ],
)
def test_invalid_augmentation_values_are_rejected(field, value):
    preprocessing = {
        "use_augmentation": True,
        "augmentation": deepcopy(AUGMENTATION),
    }
    preprocessing["augmentation"][field] = value

    with pytest.raises(ValueError):
        validate_augmentation_config(preprocessing)


def test_invalid_augmentation_scale_is_rejected():
    preprocessing = {
        "use_augmentation": True,
        "augmentation": {**AUGMENTATION, "scale": [1.05, 0.95]},
    }

    with pytest.raises(ValueError, match="positive and ordered"):
        validate_augmentation_config(preprocessing)


def test_unsupported_monitor_is_rejected(tmp_path):
    config = load_config("configs/default.yaml")
    config["training"]["monitor"] = "validation_loss"
    config_path = tmp_path / "default.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training.monitor"):
        load_config(config_path)


def test_monitor_selection_uses_macro_average_precision_and_min_delta():
    metrics = {
        "macro_auroc": 0.90,
        "macro_average_precision": 0.31,
    }
    assert resolve_monitor_score(metrics, "macro_average_precision") == 0.31
    assert monitor_improved(0.3102, 0.31, 0.0001)
    assert not monitor_improved(0.31005, 0.31, 0.0001)

    best_score = float("-inf")
    best_epoch = None
    for epoch, score in enumerate([0.20, 0.25, 0.249, 0.27], start=1):
        if monitor_improved(score, best_score, 0.0001):
            best_score = score
            best_epoch = epoch
    assert best_epoch == 4
    assert best_score == 0.27


def test_resume_requires_matching_monitor_and_augmentation():
    preprocessing = {
        "use_augmentation": True,
        "augmentation": AUGMENTATION,
    }
    protocol = build_augmentation_protocol(preprocessing)
    checkpoint = {
        "monitor_name": "macro_average_precision",
        "best_monitor_score": 0.32,
        "augmentation_protocol": protocol,
    }

    assert resolve_resume_best_score(
        checkpoint,
        "macro_average_precision",
        protocol,
    ) == 0.32

    with pytest.raises(ValueError, match="monitor mismatch"):
        resolve_resume_best_score(checkpoint, "macro_auroc", protocol)

    with pytest.raises(ValueError, match="augmentation protocol mismatch"):
        resolve_resume_best_score(
            checkpoint,
            "macro_average_precision",
            {"enabled": False, "config": None},
        )


def test_legacy_checkpoint_only_resumes_auroc_without_augmentation():
    checkpoint = {
        "best_macro_auroc": 0.88,
        "config": {"preprocessing": {"use_augmentation": False}},
    }
    disabled = {"enabled": False, "config": None}

    assert resolve_resume_best_score(
        checkpoint,
        "macro_auroc",
        disabled,
    ) == 0.88

    with pytest.raises(ValueError, match="Legacy checkpoint"):
        resolve_resume_best_score(
            checkpoint,
            "macro_average_precision",
            disabled,
        )
