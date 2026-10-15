import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.utils.config import load_config


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]

    manifest_path = Path(
        data_config["manifest_path"]
    )

    split_dir = Path(
        data_config["split_dir"]
    )

    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"

    if (
        train_path.exists()
        or val_path.exists()
    ) and not args.overwrite:
        raise FileExistsError(
            "Split already exists. Reuse it for all "
            "three backbones."
        )

    manifest = pd.read_csv(manifest_path)
    findings = data_config["target_findings"]

    if not manifest["image_id"].is_unique:
        raise ValueError(
            "Manifest contains duplicate image IDs"
        )

    labels = manifest[
        findings
    ].to_numpy(dtype=np.uint8)

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=data_config["val_fraction"],
        random_state=config["seed"],
    )

    train_index, val_index = next(
        splitter.split(
            np.zeros((len(manifest), 1)),
            labels,
        )
    )

    train_table = manifest.iloc[
        train_index
    ].copy()

    val_table = manifest.iloc[
        val_index
    ].copy()

    if not set(
        train_table["image_id"]
    ).isdisjoint(
        val_table["image_id"]
    ):
        raise ValueError(
            "Image leakage between train and validation"
        )

    summary_rows = []

    for split_name, table in [
        ("train", train_table),
        ("val", val_table),
    ]:
        for finding in findings:
            positive = int(table[finding].sum())
            negative = len(table) - positive

            if positive == 0 or negative == 0:
                raise ValueError(
                    f"{split_name}/{finding} does not "
                    "contain both classes"
                )

            summary_rows.append({
                "split": split_name,
                "finding": finding,
                "positive": positive,
                "negative": negative,
                "positive_rate": (
                    positive / len(table)
                ),
            })

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_table.sort_values(
        "image_id"
    ).to_csv(
        train_path,
        index=False,
    )

    val_table.sort_values(
        "image_id"
    ).to_csv(
        val_path,
        index=False,
    )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        split_dir / "summary.csv",
        index=False,
    )

    split_metadata = {
        "seed": config["seed"],
        "method": ("MultilabelStratifiedShuffleSplit"),
        "val_fraction": data_config["val_fraction"],
        "manifest_sha256": file_sha256(manifest_path),
        "train_images": len(train_table),
        "val_images": len(val_table),
        "target_findings": findings,
    }

    (
        split_dir / "split_metadata.json"
    ).write_text(
        json.dumps(
            split_metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()