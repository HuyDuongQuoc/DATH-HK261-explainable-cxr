from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.preprocess import project_split_manifests
from src.data.splitting import (
    SPLIT_NAMES,
    assignment_sha256,
    build_split_metadata,
    build_summary,
    create_split_tables,
    write_split_artifacts,
)


FINDINGS = ["finding_a", "finding_b", "finding_c", "finding_d"]
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def make_manifest(size=200):
    index = np.arange(size)
    return pd.DataFrame(
        {
            "image_id": [f"image_{value:04d}" for value in index],
            "raw_image_path": [f"train/image_{value:04d}.jpg" for value in index],
            "finding_a": (index % 2 == 0).astype(np.uint8),
            "finding_b": (index % 3 == 0).astype(np.uint8),
            "finding_c": (index % 5 == 0).astype(np.uint8),
            "finding_d": (index % 7 == 0).astype(np.uint8),
        }
    )


def test_split_is_exact_disjoint_complete_and_reproducible():
    manifest = make_manifest()
    first, sizes = create_split_tables(manifest, FINDINGS, FRACTIONS, seed=42)
    second, _ = create_split_tables(manifest, FINDINGS, FRACTIONS, seed=42)

    assert sizes == {"train": 140, "val": 30, "test": 30}
    assert {name: len(first[name]) for name in SPLIT_NAMES} == sizes
    assert assignment_sha256(first) == assignment_sha256(second)

    id_sets = {
        name: set(first[name]["image_id"]) for name in SPLIT_NAMES
    }
    assert id_sets["train"].isdisjoint(id_sets["val"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["val"].isdisjoint(id_sets["test"])
    assert set().union(*id_sets.values()) == set(manifest["image_id"])
    for table in first.values():
        for finding in FINDINGS:
            assert 0 < int(table[finding].sum()) < len(table)


def test_split_artifacts_require_overwrite(tmp_path):
    manifest = make_manifest()
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    tables, sizes = create_split_tables(manifest, FINDINGS, FRACTIONS, seed=42)
    summary = build_summary(tables, FINDINGS)
    metadata = build_split_metadata(
        manifest_path,
        tables,
        FINDINGS,
        FRACTIONS,
        sizes,
        seed=42,
    )
    split_dir = tmp_path / "splits"

    write_split_artifacts(split_dir, tables, summary, metadata)
    with pytest.raises(FileExistsError):
        write_split_artifacts(split_dir, tables, summary, metadata)
    write_split_artifacts(
        split_dir,
        tables,
        summary,
        metadata,
        overwrite=True,
    )

    assert len(pd.read_csv(split_dir / "summary.csv")) == 12
    assert all((split_dir / f"{name}.csv").is_file() for name in SPLIT_NAMES)
    assert (split_dir / "split_metadata.json").is_file()


def test_project_processed_splits_without_images(tmp_path):
    manifest = make_manifest()
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    tables, sizes = create_split_tables(manifest, FINDINGS, FRACTIONS, seed=42)
    raw_split_dir = tmp_path / "raw_splits"
    metadata = build_split_metadata(
        manifest_path,
        tables,
        FINDINGS,
        FRACTIONS,
        sizes,
        seed=42,
    )
    write_split_artifacts(
        raw_split_dir,
        tables,
        build_summary(tables, FINDINGS),
        metadata,
    )

    processed = manifest[["image_id", *FINDINGS]].copy()
    processed["image_path"] = processed["image_id"].map(
        lambda value: f"images/{value}.png"
    )
    processed["mask_path"] = processed["image_id"].map(
        lambda value: f"masks/{value}.npz"
    )
    processed_path = tmp_path / "processed_manifest.csv"
    processed.to_csv(processed_path, index=False)

    output_dir = tmp_path / "processed_splits"
    counts = project_split_manifests(
        processed_manifest_path=processed_path,
        raw_split_dir=raw_split_dir,
        processed_split_dir=output_dir,
    )
    assert counts == sizes
    assert (output_dir / "summary.csv").read_bytes() == (
        raw_split_dir / "summary.csv"
    ).read_bytes()
    assert (output_dir / "split_metadata.json").read_bytes() == (
        raw_split_dir / "split_metadata.json"
    ).read_bytes()


def test_current_manifest_protocol_when_available():
    manifest_path = Path("data/processed/manifest.csv")
    if not manifest_path.is_file():
        pytest.skip("Real manifest is not available")

    manifest = pd.read_csv(manifest_path)
    findings = [
        "Aortic enlargement",
        "Atelectasis",
        "Calcification",
        "Cardiomegaly",
        "Consolidation",
        "ILD",
        "Infiltration",
        "Lung Opacity",
        "Nodule/Mass",
        "Other lesion",
        "Pleural effusion",
        "Pleural thickening",
        "Pneumothorax",
        "Pulmonary fibrosis",
    ]
    tables, sizes = create_split_tables(
        manifest,
        findings,
        FRACTIONS,
        seed=42,
    )

    assert sizes == {"train": 10_500, "val": 2_250, "test": 2_250}
    assert [int(tables[name]["Atelectasis"].sum()) for name in SPLIT_NAMES] == [
        43,
        10,
        9,
    ]
    assert [int(tables[name]["Pneumothorax"].sum()) for name in SPLIT_NAMES] == [
        41,
        8,
        9,
    ]
