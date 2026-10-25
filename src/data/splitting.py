import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


SPLIT_NAMES = ("train", "val", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    values = np.asarray(
        [fractions[name] for name in SPLIT_NAMES],
        dtype=float,
    )
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Split fractions must be finite and positive")
    if not np.isclose(values.sum(), 1.0):
        raise ValueError("Split fractions must sum to 1")

    raw_sizes = values * total
    rounded_sizes = np.rint(raw_sizes).astype(int)
    if not np.allclose(raw_sizes, rounded_sizes):
        raise ValueError(
            "Split fractions do not produce integer sizes for this manifest"
        )
    if rounded_sizes.sum() != total:
        raise ValueError("Rounded split sizes do not cover the manifest")
    return dict(zip(SPLIT_NAMES, rounded_sizes.tolist()))


def validate_split_tables(manifest, split_tables, findings, expected_sizes):
    manifest_ids = set(manifest["image_id"].astype(str))
    split_id_sets = {}

    for split_name in SPLIT_NAMES:
        if split_name not in split_tables:
            raise ValueError(f"Missing split table: {split_name}")
        table = split_tables[split_name]
        if len(table) != expected_sizes[split_name]:
            raise ValueError(
                f"Unexpected {split_name} size: "
                f"{len(table)} != {expected_sizes[split_name]}"
            )
        if not table["image_id"].is_unique:
            raise ValueError(f"Duplicate image IDs in {split_name}")
        split_id_sets[split_name] = set(table["image_id"].astype(str))

        for finding in findings:
            positive = int(table[finding].sum())
            if positive == 0 or positive == len(table):
                raise ValueError(
                    f"{split_name}/{finding} does not contain both classes"
                )

    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            if not split_id_sets[left].isdisjoint(split_id_sets[right]):
                raise ValueError(f"Image leakage between {left} and {right}")

    combined = set().union(*split_id_sets.values())
    if combined != manifest_ids:
        raise ValueError("Split IDs do not exactly cover the manifest")


def create_split_tables(manifest, findings, fractions, seed):
    if not manifest["image_id"].is_unique:
        raise ValueError("Manifest contains duplicate image IDs")

    missing = set(findings) - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing findings: {sorted(missing)}")

    labels = manifest[findings].to_numpy(dtype=np.uint8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Manifest labels must be binary")

    sizes = expected_split_sizes(len(manifest), fractions)
    holdout_fraction = fractions["val"] + fractions["test"]
    test_share_of_holdout = fractions["test"] / holdout_fraction
    placeholder = np.zeros((len(manifest), 1), dtype=np.uint8)

    first_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=holdout_fraction,
        random_state=seed,
    )
    train_index, holdout_index = next(
        first_splitter.split(placeholder, labels)
    )

    second_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_share_of_holdout,
        random_state=seed,
    )
    val_relative, test_relative = next(
        second_splitter.split(
            np.zeros((len(holdout_index), 1), dtype=np.uint8),
            labels[holdout_index],
        )
    )

    split_tables = {
        "train": manifest.iloc[train_index].copy(),
        "val": manifest.iloc[holdout_index[val_relative]].copy(),
        "test": manifest.iloc[holdout_index[test_relative]].copy(),
    }
    validate_split_tables(
        manifest=manifest,
        split_tables=split_tables,
        findings=findings,
        expected_sizes=sizes,
    )
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
    split_tables,
    findings,
    fractions,
    sizes,
    seed,
):
    all_zero = {}
    for split_name, table in split_tables.items():
        zero_count = int(table[findings].sum(axis=1).eq(0).sum())
        all_zero[split_name] = {
            "count": zero_count,
            "rate": zero_count / len(table),
        }

    return {
        "seed": seed,
        "method": "two_stage_MultilabelStratifiedShuffleSplit",
        "iterative_stratification_version": version(
            "iterative-stratification"
        ),
        "fractions": fractions,
        "expected_images": sizes,
        "actual_images": {
            name: len(split_tables[name]) for name in SPLIT_NAMES
        },
        "all_zero": all_zero,
        "manifest_sha256": file_sha256(manifest_path),
        "assignment_sha256": assignment_sha256(split_tables),
        "target_findings": list(findings),
    }


def write_split_artifacts(
    split_dir,
    split_tables,
    summary,
    metadata,
    overwrite=False,
):
    split_dir = Path(split_dir)
    outputs = {
        **{name: split_dir / f"{name}.csv" for name in SPLIT_NAMES},
        "summary": split_dir / "summary.csv",
        "metadata": split_dir / "split_metadata.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Split artifacts already exist: "
            + ", ".join(str(path) for path in existing)
        )

    split_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {}
    try:
        for split_name in SPLIT_NAMES:
            temporary = outputs[split_name].with_suffix(".csv.tmp")
            split_tables[split_name].sort_values("image_id").to_csv(
                temporary,
                index=False,
            )
            temporary_paths[split_name] = temporary

        summary_temporary = outputs["summary"].with_suffix(".csv.tmp")
        summary.to_csv(summary_temporary, index=False)
        temporary_paths["summary"] = summary_temporary

        metadata_temporary = outputs["metadata"].with_suffix(".json.tmp")
        metadata_temporary.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_paths["metadata"] = metadata_temporary

        for name, temporary in temporary_paths.items():
            temporary.replace(outputs[name])
    finally:
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()
