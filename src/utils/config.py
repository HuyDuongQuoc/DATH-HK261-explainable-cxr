from copy import deepcopy
import math
from pathlib import Path
import yaml


SUPPORTED_MONITORS = {
    "macro_auroc",
    "macro_average_precision",
}

SUPPORTED_LOSSES = {"weighted_bce", "sigmoid_focal"}
SUPPORTED_POSITIVE_WEIGHT_MODES = {
    "raw_ratio",
    "sqrt_ratio",
    "sqrt_ratio_capped",
}


def merge_dict(base: dict, override: dict) -> dict:
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value

    return result


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_augmentation_config(preprocessing):
    enabled = preprocessing.get("use_augmentation")
    if not isinstance(enabled, bool):
        raise ValueError("preprocessing.use_augmentation must be boolean")

    augmentation = preprocessing.get("augmentation")
    if not isinstance(augmentation, dict):
        raise ValueError("preprocessing.augmentation must be a mapping")

    rotation = _finite_number(
        augmentation.get("rotation_degrees"),
        "preprocessing.augmentation.rotation_degrees",
    )
    if not 0.0 <= rotation <= 30.0:
        raise ValueError("rotation_degrees must be in [0, 30]")

    translate = _finite_number(
        augmentation.get("translate_fraction"),
        "preprocessing.augmentation.translate_fraction",
    )
    if not 0.0 <= translate <= 0.25:
        raise ValueError("translate_fraction must be in [0, 0.25]")

    scale = augmentation.get("scale")
    if not isinstance(scale, list) or len(scale) != 2:
        raise ValueError("preprocessing.augmentation.scale must contain two values")
    scale_min = _finite_number(scale[0], "preprocessing.augmentation.scale[0]")
    scale_max = _finite_number(scale[1], "preprocessing.augmentation.scale[1]")
    if scale_min <= 0.0 or scale_max <= 0.0 or scale_min > scale_max:
        raise ValueError("augmentation scale must be positive and ordered")

    probability = _finite_number(
        augmentation.get("intensity_probability"),
        "preprocessing.augmentation.intensity_probability",
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError("intensity_probability must be in [0, 1]")

    for name in ("brightness", "contrast"):
        value = _finite_number(
            augmentation.get(name),
            f"preprocessing.augmentation.{name}",
        )
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


def validate_loss_config(training):
    loss = training.get("loss")
    if loss is None:
        loss = {
            "name": "weighted_bce",
            "positive_weight": {
                "mode": "raw_ratio",
                "max_value": None,
            },
        }
        training["loss"] = loss
    if not isinstance(loss, dict):
        raise ValueError("training.loss must be a mapping")
    if loss.get("name") not in SUPPORTED_LOSSES:
        raise ValueError(
            "training.loss.name must be one of "
            f"{sorted(SUPPORTED_LOSSES)}"
        )

    gamma = loss.get("gamma")
    if loss["name"] == "sigmoid_focal":
        gamma = _finite_number(gamma, "training.loss.gamma")
        if not 0.0 < gamma <= 5.0:
            raise ValueError("training.loss.gamma must be in (0, 5]")
        loss["gamma"] = gamma
    elif "gamma" in loss:
        raise ValueError("training.loss.gamma is not accepted for weighted_bce")

    positive_weight = loss.get("positive_weight")
    if not isinstance(positive_weight, dict):
        raise ValueError("training.loss.positive_weight must be a mapping")
    mode = positive_weight.get("mode")
    if mode not in SUPPORTED_POSITIVE_WEIGHT_MODES:
        raise ValueError(
            "training.loss.positive_weight.mode must be one of "
            f"{sorted(SUPPORTED_POSITIVE_WEIGHT_MODES)}"
        )
    maximum = positive_weight.get("max_value")
    if mode == "sqrt_ratio_capped":
        maximum = _finite_number(
            maximum,
            "training.loss.positive_weight.max_value",
        )
        if maximum <= 0.0:
            raise ValueError("positive weight max_value must be positive")
        positive_weight["max_value"] = maximum
    elif maximum is not None:
        raise ValueError(
            "positive weight max_value must be null unless mode is "
            "sqrt_ratio_capped"
        )


def validate_training_config(training):
    for field in ("batch_size", "validation_batch_size", "num_workers"):
        value = training.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"training.{field} must be an integer")
        minimum = 0 if field == "num_workers" else 1
        if value < minimum:
            raise ValueError(f"training.{field} must be >= {minimum}")
    validate_loss_config(training)


def load_config(config_path):
    config_path = Path(config_path)
    default_path = config_path.parent / "default.yaml"

    if not default_path.is_file():
        raise FileNotFoundError(default_path)

    with default_path.open("r", encoding="utf-8") as file:
        default_config = yaml.safe_load(file)

    if config_path.name == "default.yaml":
        config = default_config
    else:
        if not config_path.is_file():
            raise FileNotFoundError(config_path)

        with config_path.open("r", encoding="utf-8") as file:
            model_config = yaml.safe_load(file)

        config = merge_dict(default_config, model_config)

    findings = config["data"]["target_findings"]

    if not findings:
        raise ValueError(
            "target_findings is empty. "
            "Select classes from the EDA result first."
        )

    if len(findings) != len(set(findings)):
        raise ValueError("target_findings contains duplicates")

    data = config["data"]
    fractions = (
        float(data["train_fraction"]),
        float(data["val_fraction"]),
    )
    if any(value <= 0 for value in fractions):
        raise ValueError("train/val fractions must be positive")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("train/val fractions must sum to 1")
    if not data.get("protocol_id"):
        raise ValueError("data.protocol_id is required")

    validate_augmentation_config(config["preprocessing"])

    training = config["training"]
    training.setdefault("validation_batch_size", training["batch_size"])
    validate_training_config(training)

    monitor = training.get("monitor")
    if monitor not in SUPPORTED_MONITORS:
        raise ValueError(
            "training.monitor must be one of "
            f"{sorted(SUPPORTED_MONITORS)}"
        )

    return config
