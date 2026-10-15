from copy import deepcopy
from pathlib import Path
import yaml


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

    return config