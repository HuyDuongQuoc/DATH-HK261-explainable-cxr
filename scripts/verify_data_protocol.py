import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.annotations import read_annotations, validate_bbox
from src.data.protocol_splitting import assignment_sha256
from src.utils.checksums import file_sha256, protocol_file_hashes
from src.utils.config import load_config


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_manifest(path, findings, expected_count):
    table = pd.read_csv(path, dtype={"image_id": str})
    require(len(table) == expected_count, f"Unexpected row count for {path}: {len(table)}")
    require(table["image_id"].is_unique, f"Duplicate image IDs in {path}")
    require(set(findings).issubset(table.columns), f"Missing finding columns in {path}")
    labels = table[findings].to_numpy()
    require(np.isin(labels, [0, 1]).all(), f"Non-binary labels in {path}")
    for finding in findings:
        positive = int(table[finding].sum())
        require(0 < positive < len(table), f"{path}/{finding} lacks both classes")
    return table


def verify_materialized_samples(table, root, findings, expected_binary_masks):
    root = Path(root)
    for _, row in table.iterrows():
        image_path = root / row["image_path"]
        mask_path = root / row["mask_path"]
        require(image_path.is_file(), f"Missing processed image: {image_path}")
        require(mask_path.is_file(), f"Missing processed mask: {mask_path}")
        with np.load(mask_path, allow_pickle=False) as stored:
            require(stored["class_names"].tolist() == list(findings), f"Class order mismatch: {mask_path}")
            labels = row[findings].to_numpy(dtype=np.uint8)
            require(np.array_equal(stored["labels"].astype(np.uint8), labels), f"Label mismatch: {mask_path}")
            expected_shape = (len(findings), 224, 224)
            require(tuple(stored["soft_masks"].shape) == expected_shape, f"Soft-mask shape: {mask_path}")
            require(tuple(stored["union_masks"].shape) == expected_shape, f"Union-mask shape: {mask_path}")
            if expected_binary_masks:
                require(np.isin(stored["soft_masks"], [0, 1]).all(), f"Non-binary test soft mask: {mask_path}")
                require(np.isin(stored["union_masks"], [0, 1]).all(), f"Non-binary test union mask: {mask_path}")


def verify_external_boxes(annotation_path, image_dir):
    annotations = read_annotations(annotation_path)
    for row in annotations.dropna(subset=["x_min", "y_min", "x_max", "y_max"]).itertuples(index=False):
        image_path = Path(image_dir) / f"{row.image_id}.jpg"
        with Image.open(image_path) as image:
            width, height = image.size
        validate_bbox(row, width=width, height=height)
    return annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    external = data["external_test"]
    findings = list(data["target_findings"])
    protocol_dir = Path(data["protocol_dir"])

    validation_ids = pd.read_csv(data["validation_ids_path"], dtype={"image_id": str})
    assignments = pd.read_csv(protocol_dir / "split_assignments.csv", dtype={"image_id": str})
    split_metadata = json.loads((protocol_dir / "split_metadata.json").read_text(encoding="utf-8"))
    external_metadata = json.loads(
        (protocol_dir / "external_test_metadata.json").read_text(encoding="utf-8")
    )
    train = load_manifest(data["train_manifest_path"], findings, 12_750)
    val = load_manifest(data["val_manifest_path"], findings, 2_250)
    test = load_manifest(external["manifest_path"], findings, 3_000)

    train_ids = set(train["image_id"])
    val_ids = set(val["image_id"])
    test_ids = set(test["image_id"])
    require(train_ids.isdisjoint(val_ids), "Train/validation overlap")
    require(train_ids.isdisjoint(test_ids), "Train/external-test overlap")
    require(val_ids.isdisjoint(test_ids), "Validation/external-test overlap")
    require(set(validation_ids["image_id"]) == val_ids, "Validation IDs changed")
    require(len(train_ids | val_ids) == 15_000, "Train/validation do not cover 15,000 images")
    require(set(assignments["split"]) == {"train", "val"}, "Unexpected split names")
    require(len(assignments) == 15_000, "Unexpected assignment count")

    split_tables = {
        "train": assignments.loc[assignments["split"].eq("train"), ["image_id"]],
        "val": assignments.loc[assignments["split"].eq("val"), ["image_id"]],
    }
    require(split_metadata["protocol_id"] == data["protocol_id"], "Split protocol ID mismatch")
    require(split_metadata["assignment_sha256"] == assignment_sha256(split_tables), "Assignment hash mismatch")
    require(split_metadata["manifest_sha256"] == file_sha256(data["manifest_path"]), "Manifest hash mismatch")
    require(
        split_metadata["validation_ids_sha256"] == file_sha256(data["validation_ids_path"]),
        "Validation-ID hash mismatch",
    )

    raw_test_ids = {path.stem for path in Path(external["image_dir"]).glob("*.jpg")}
    require(raw_test_ids == test_ids, "External-test CSV/JPG set mismatch")
    annotations = verify_external_boxes(external["annotation_csv"], external["image_dir"])
    require(external_metadata["protocol_id"] == data["protocol_id"], "External protocol ID mismatch")
    require(external_metadata["annotation_sha256"] == file_sha256(external["annotation_csv"]), "Annotation hash mismatch")
    require(external_metadata["manifest_sha256"] == file_sha256(external["manifest_path"]), "Test manifest hash mismatch")
    require(int(annotations["image_id"].nunique()) == 3_000, "Unexpected annotated test images")

    verify_materialized_samples(train, data["processed_root"], findings, expected_binary_masks=False)
    verify_materialized_samples(val, data["processed_root"], findings, expected_binary_masks=False)
    verify_materialized_samples(test, external["processed_root"], findings, expected_binary_masks=True)

    expected_hashes = json.loads((protocol_dir / "checksums.json").read_text(encoding="utf-8"))
    require(expected_hashes == protocol_file_hashes(protocol_dir), "Protocol checksums are stale")

    result = {
        "status": "success",
        "protocol_id": data["protocol_id"],
        "train_images": len(train),
        "val_images": len(val),
        "external_test_images": len(test),
        "assignment_sha256": split_metadata["assignment_sha256"],
        "external_test_manifest_sha256": external_metadata["manifest_sha256"],
        "overlap": 0,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
