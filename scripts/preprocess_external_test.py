import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.annotations import NORMAL_CLASS, get_reader_roster, read_annotations
from src.data.preprocessing import preprocess_image
from src.utils.checksums import file_sha256, write_protocol_checksums
from src.utils.config import load_config


def image_ids_sha256(image_ids):
    payload = json.dumps(sorted(str(value) for value in image_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_external_source(annotations, image_dir, findings, reader_id, train_ids):
    image_dir = Path(image_dir)
    annotation_ids = set(annotations["image_id"].astype(str))
    image_ids = {path.stem for path in image_dir.glob("*.jpg")}
    if annotation_ids != image_ids:
        raise ValueError(
            "External-test annotations/images mismatch: "
            f"missing_images={len(annotation_ids - image_ids)}, "
            f"missing_annotations={len(image_ids - annotation_ids)}"
        )
    overlap = annotation_ids & set(str(value) for value in train_ids)
    if overlap:
        raise ValueError(f"External test overlaps train corpus: {sorted(overlap)[:5]}")
    expected_classes = set(findings) | {NORMAL_CLASS}
    actual_classes = set(annotations["class_name"])
    if actual_classes != expected_classes:
        raise ValueError(
            f"External-test class set mismatch: {sorted(actual_classes ^ expected_classes)}"
        )
    if set(annotations["rad_id"]) != {reader_id}:
        raise ValueError("External test must contain exactly the configured consensus reader")
    normal_ids = set(
        annotations.loc[annotations["class_name"].eq(NORMAL_CLASS), "image_id"].astype(str)
    )
    abnormal_ids = set(
        annotations.loc[~annotations["class_name"].eq(NORMAL_CLASS), "image_id"].astype(str)
    )
    if normal_ids & abnormal_ids:
        raise ValueError("External-test images cannot be both normal and abnormal")
    return annotation_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    external = data["external_test"]
    preprocessing = config["preprocessing"]
    findings = list(data["target_findings"])

    annotation_path = Path(external["annotation_csv"])
    image_dir = Path(external["image_dir"])
    processed_root = Path(external["processed_root"])
    manifest_path = Path(external["manifest_path"])
    metadata_path = Path(data["protocol_dir"]) / "external_test_metadata.json"
    image_output_dir = processed_root / "images"
    mask_output_dir = processed_root / "masks"

    guarded_outputs = [manifest_path, metadata_path, processed_root / "metadata.json"]
    existing = [path for path in guarded_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "External-test artifacts already exist: " + ", ".join(str(path) for path in existing)
        )

    annotations = read_annotations(annotation_path)
    train_manifest = pd.read_csv(data["manifest_path"], dtype={"image_id": str})
    annotation_ids = validate_external_source(
        annotations=annotations,
        image_dir=image_dir,
        findings=findings,
        reader_id=external["reader_id"],
        train_ids=train_manifest["image_id"],
    )
    roster = get_reader_roster(annotations, expected_readers=1)
    groups = {str(image_id): group for image_id, group in annotations.groupby("image_id")}

    image_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, image_id in enumerate(sorted(annotation_ids), start=1):
        result = preprocess_image(
            image_path=image_dir / f"{image_id}.jpg",
            annotations_for_image=groups[image_id],
            findings=findings,
            readers=roster[image_id],
            image_size=preprocessing["image_size"],
            positive_votes=1,
        )
        output_image_path = image_output_dir / f"{image_id}.png"
        output_mask_path = mask_output_dir / f"{image_id}.npz"
        result["image"].save(output_image_path)
        np.savez_compressed(
            output_mask_path,
            # External ground truth has one consensus reader. Preserve it as a
            # binary spatial target after resizing instead of introducing
            # fractional edge coverage through area interpolation.
            soft_masks=result["union_masks"].astype(np.float16),
            union_masks=result["union_masks"].astype(np.uint8),
            labels=result["labels"].astype(np.uint8),
            mask_valid=result["mask_valid"].astype(bool),
            positive_reader_counts=result["positive_reader_counts"].astype(np.uint8),
            class_names=np.asarray(findings, dtype=np.str_),
            num_readers=np.asarray(result["num_readers"], dtype=np.uint8),
        )
        row = {
            "image_id": image_id,
            "image_path": f"images/{image_id}.png",
            "mask_path": f"masks/{image_id}.npz",
        }
        row.update(
            {finding: int(result["labels"][class_index]) for class_index, finding in enumerate(findings)}
        )
        rows.append(row)
        if index % 500 == 0:
            print(f"Processed external test {index}/{len(annotation_ids)}", flush=True)

    manifest = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    positive_counts = {finding: int(manifest[finding].sum()) for finding in findings}
    metadata = {
        "protocol_id": data["protocol_id"],
        "dataset": "external_test",
        "annotation_rule": "single_consensus_reader",
        "mask_rule": "binary_resized_consensus",
        "reader_id": external["reader_id"],
        "positive_votes": 1,
        "num_images": len(manifest),
        "normal_images": int(manifest[findings].sum(axis=1).eq(0).sum()),
        "abnormal_images": int(manifest[findings].sum(axis=1).gt(0).sum()),
        "bbox_rows": int(annotations["x_min"].notna().sum()),
        "target_findings": findings,
        "positive_images": positive_counts,
        "annotation_sha256": file_sha256(annotation_path),
        "train_manifest_sha256": file_sha256(data["manifest_path"]),
        "manifest_sha256": file_sha256(manifest_path),
        "image_ids_sha256": image_ids_sha256(annotation_ids),
        "image_size": preprocessing["image_size"],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (processed_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_protocol_checksums(data["protocol_dir"])
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
