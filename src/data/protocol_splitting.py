import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.checksums import file_sha256, write_protocol_checksums


SPLIT_NAMES = ("train", "val")


def assignment_sha256(split_tables):
    assignments = sorted(
        (str(image_id), split_name)
        for split_name, table in split_tables.items()
        for image_id in table["image_id"]
    )
    payload = json.dumps(
        assignments,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_split_sizes(total, fractions):
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"Expected fractions for {SPLIT_NAMES}")
    values = np.asarray([fractions[name] for name in SPLIT_NAMES], dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Split fractions must be finite and positive")
    if not np.isclose(values.sum(), 1.0):
        raise ValueError("Split fractions must sum to 1")
    raw_sizes = values * total
    rounded_sizes = np.rint(raw_sizes).astype(int)
    if not np.allclose(raw_sizes, rounded_sizes):
        raise ValueError("Split fractions do not produce integer sizes for this manifest")
    return dict(zip(SPLIT_NAMES, rounded_sizes.tolist()))


def load_validation_ids(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, dtype={"image_id": str})
    if list(table.columns) != ["image_id"]:
        raise ValueError("validation_ids.csv must contain only image_id")
    if table.empty or not table["image_id"].is_unique:
        raise ValueError("Validation IDs must be non-empty and unique")
    return table["image_id"].astype(str).tolist()


def validate_split_tables(manifest, split_tables, findings, expected_sizes):
    manifest_ids = set(manifest["image_id"].astype(str))
    split_id_sets = {}
    for split_name in SPLIT_NAMES:
        if split_name not in split_tables:
            raise ValueError(f"Missing split table: {split_name}")
        table = split_tables[split_name]
        if len(table) != expected_sizes[split_name]:
            raise ValueError(
                f"Unexpected {split_name} size: {len(table)} != {expected_sizes[split_name]}"
            )
        if not table["image_id"].is_unique:
            raise ValueError(f"Duplicate image IDs in {split_name}")
        split_id_sets[split_name] = set(table["image_id"].astype(str))
        for finding in findings:
            positive = int(table[finding].sum())
            if positive == 0 or positive == len(table):
                raise ValueError(f"{split_name}/{finding} does not contain both classes")
    if not split_id_sets["train"].isdisjoint(split_id_sets["val"]):
        raise ValueError("Image leakage between train and val")
    if set().union(*split_id_sets.values()) != manifest_ids:
        raise ValueError("Split IDs do not exactly cover the manifest")


def create_fixed_split_tables(manifest, findings, validation_ids, fractions):
    manifest = manifest.copy()
    manifest["image_id"] = manifest["image_id"].astype(str)
    if not manifest["image_id"].is_unique:
        raise ValueError("Manifest contains duplicate image IDs")
    missing = set(findings) - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing findings: {sorted(missing)}")
    labels = manifest[findings].to_numpy(dtype=np.uint8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Manifest labels must be binary")
    sizes = expected_split_sizes(len(manifest), fractions)
    validation_ids = [str(value) for value in validation_ids]
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Validation IDs contain duplicates")
    if len(validation_ids) != sizes["val"]:
        raise ValueError(
            f"Unexpected validation ID count: {len(validation_ids)} != {sizes['val']}"
        )
    manifest_ids = set(manifest["image_id"])
    unknown = set(validation_ids) - manifest_ids
    if unknown:
        raise ValueError(f"Unknown validation IDs: {sorted(unknown)[:5]}")
    is_val = manifest["image_id"].isin(set(validation_ids))
    split_tables = {
        "train": manifest.loc[~is_val].copy(),
        "val": manifest.loc[is_val].copy(),
    }
    validate_split_tables(manifest, split_tables, findings, sizes)
    return split_tables, sizes


def build_summary(split_tables, findings):
    rows = []
    for split_name in SPLIT_NAMES:
        table = split_tables[split_name]
        for finding in findings:
            positive = int(table[finding].sum())
            rows.append(
                {
                    "split": split_name,
                    "finding": finding,
                    "positive": positive,
                    "negative": len(table) - positive,
                    "positive_rate": positive / len(table),
                }
            )
    return pd.DataFrame(rows)


def build_split_metadata(
    manifest_path,
    validation_ids_path,
    split_tables,
    findings,
    fractions,
    sizes,
    protocol_id,
):
    all_zero = {}
    for split_name, table in split_tables.items():
        zero_count = int(table[findings].sum(axis=1).eq(0).sum())
        all_zero[split_name] = {"count": zero_count, "rate": zero_count / len(table)}
    return {
        "protocol_id": protocol_id,
        "method": "fixed_validation_ids_complement",
        "fractions": fractions,
        "expected_images": sizes,
        "actual_images": {name: len(split_tables[name]) for name in SPLIT_NAMES},
        "all_zero": all_zero,
        "manifest_sha256": file_sha256(manifest_path),
        "validation_ids_sha256": file_sha256(validation_ids_path),
        "assignment_sha256": assignment_sha256(split_tables),
        "target_findings": list(findings),
    }


def write_split_artifacts(protocol_dir, split_tables, summary, metadata, overwrite=False):
    protocol_dir = Path(protocol_dir)
    outputs = {
        "assignments": protocol_dir / "split_assignments.csv",
        "summary": protocol_dir / "split_summary.csv",
        "metadata": protocol_dir / "split_metadata.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Protocol split artifacts already exist: " + ", ".join(str(path) for path in existing)
        )
    protocol_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.concat(
        [
            table[["image_id"]].assign(split=split_name)
            for split_name, table in split_tables.items()
        ],
        ignore_index=True,
    ).sort_values("image_id")
    assignments.to_csv(outputs["assignments"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    outputs["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_protocol_checksums(protocol_dir)
