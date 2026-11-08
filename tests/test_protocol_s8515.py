import json

import numpy as np
import pandas as pd
import pytest

from scripts.make_splits import bootstrap_validation_ids
from scripts.preprocess import project_protocol_manifests
from src.data.protocol_splitting import (
    assignment_sha256,
    build_split_metadata,
    build_summary,
    create_fixed_split_tables,
    load_validation_ids,
    write_split_artifacts,
)


FINDINGS = ["finding_a", "finding_b"]
FRACTIONS = {"train": 0.85, "val": 0.15}


def make_manifest(size=100):
    index = np.arange(size)
    return pd.DataFrame(
        {
            "image_id": [f"image_{value:03d}" for value in index],
            "raw_image_path": [f"train/image_{value:03d}.jpg" for value in index],
            "finding_a": (index % 2 == 0).astype(np.uint8),
            "finding_b": (index % 3 == 0).astype(np.uint8),
        }
    )


def test_fixed_validation_ids_create_exact_reproducible_split(tmp_path):
    manifest = make_manifest()
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    source_val = manifest.iloc[:15]
    source_path = tmp_path / "old_val.csv"
    source_val.to_csv(source_path, index=False)
    validation_path = tmp_path / "protocol" / "validation_ids.csv"
    bootstrap_validation_ids(source_path, validation_path)

    validation_ids = load_validation_ids(validation_path)
    first, sizes = create_fixed_split_tables(manifest, FINDINGS, validation_ids, FRACTIONS)
    second, _ = create_fixed_split_tables(manifest, FINDINGS, validation_ids, FRACTIONS)

    assert sizes == {"train": 85, "val": 15}
    assert set(first["val"]["image_id"]) == set(source_val["image_id"])
    assert assignment_sha256(first) == assignment_sha256(second)
    assert set(first["train"]["image_id"]).isdisjoint(first["val"]["image_id"])

    metadata = build_split_metadata(
        manifest_path=manifest_path,
        validation_ids_path=validation_path,
        split_tables=first,
        findings=FINDINGS,
        fractions=FRACTIONS,
        sizes=sizes,
        protocol_id="test-protocol",
    )
    write_split_artifacts(
        protocol_dir=validation_path.parent,
        split_tables=first,
        summary=build_summary(first, FINDINGS),
        metadata=metadata,
    )
    stored = json.loads((validation_path.parent / "split_metadata.json").read_text())
    assert stored["protocol_id"] == "test-protocol"
    assert stored["actual_images"] == sizes


def test_bootstrap_is_one_time_and_projection_uses_assignments(tmp_path):
    manifest = make_manifest()
    old_val = tmp_path / "old_val.csv"
    manifest.iloc[:15].to_csv(old_val, index=False)
    protocol = tmp_path / "protocol"
    validation_path = protocol / "validation_ids.csv"
    bootstrap_validation_ids(old_val, validation_path)
    with pytest.raises(FileExistsError):
        bootstrap_validation_ids(old_val, validation_path)

    tables, _ = create_fixed_split_tables(
        manifest, FINDINGS, load_validation_ids(validation_path), FRACTIONS
    )
    assignments = pd.concat(
        [table[["image_id"]].assign(split=name) for name, table in tables.items()],
        ignore_index=True,
    )
    assignments_path = protocol / "split_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    processed = manifest[["image_id", *FINDINGS]].copy()
    processed["image_path"] = processed["image_id"].map(lambda value: f"images/{value}.png")
    processed["mask_path"] = processed["image_id"].map(lambda value: f"masks/{value}.npz")
    processed_path = tmp_path / "processed.csv"
    processed.to_csv(processed_path, index=False)

    counts = project_protocol_manifests(
        processed_manifest_path=processed_path,
        assignments_path=assignments_path,
        train_manifest_path=protocol / "train_manifest.csv",
        val_manifest_path=protocol / "val_manifest.csv",
        protocol_dir=protocol,
    )
    assert counts == {"train": 85, "val": 15}
    assert len(pd.read_csv(protocol / "train_manifest.csv")) == 85
    assert len(pd.read_csv(protocol / "val_manifest.csv")) == 15
