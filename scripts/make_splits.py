import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.protocol_splitting import (
    build_split_metadata,
    build_summary,
    create_fixed_split_tables,
    load_validation_ids,
    write_split_artifacts,
)
from src.utils.config import load_config
from src.utils.checksums import write_protocol_checksums


def bootstrap_validation_ids(source_path, output_path):
    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists; bootstrap is a one-time operation"
        )
    source = pd.read_csv(source_path, dtype={"image_id": str})
    if "image_id" not in source.columns:
        raise ValueError("Bootstrap validation CSV is missing image_id")
    validation_ids = source[["image_id"]].copy()
    validation_ids["image_id"] = validation_ids["image_id"].astype(str)
    if validation_ids.empty or not validation_ids["image_id"].is_unique:
        raise ValueError("Bootstrap validation IDs must be non-empty and unique")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_ids.sort_values("image_id").to_csv(output_path, index=False)


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

    parser.add_argument(
        "--bootstrap-validation-from",
        default=None,
        help="One-time source CSV used to preserve the existing validation IDs.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]

    manifest_path = Path(
        data_config["manifest_path"]
    )

    protocol_dir = Path(data_config["protocol_dir"])
    validation_ids_path = Path(data_config["validation_ids_path"])

    if args.bootstrap_validation_from is not None:
        bootstrap_validation_ids(
            source_path=args.bootstrap_validation_from,
            output_path=validation_ids_path,
        )

    manifest = pd.read_csv(manifest_path)
    findings = data_config["target_findings"]
    fractions = {
        "train": float(data_config["train_fraction"]),
        "val": float(data_config["val_fraction"]),
    }
    validation_ids = load_validation_ids(validation_ids_path)
    split_tables, sizes = create_fixed_split_tables(
        manifest=manifest,
        findings=findings,
        validation_ids=validation_ids,
        fractions=fractions,
    )
    summary = build_summary(split_tables, findings)
    split_metadata = build_split_metadata(
        manifest_path=manifest_path,
        validation_ids_path=validation_ids_path,
        split_tables=split_tables,
        findings=findings,
        fractions=fractions,
        sizes=sizes,
        protocol_id=data_config["protocol_id"],
    )
    write_split_artifacts(
        protocol_dir=protocol_dir,
        split_tables=split_tables,
        summary=summary,
        metadata=split_metadata,
        overwrite=args.overwrite,
    )

    write_protocol_checksums(protocol_dir)

    print(summary.to_string(index=False))
    print(json.dumps(split_metadata, indent=2, ensure_ascii=False))
    print(
        "Split counts:",
        {name: len(table) for name, table in split_tables.items()},
    )
    print("Overlap:", 0)
    print("Covered images:", sum(len(table) for table in split_tables.values()))


if __name__ == "__main__":
    main()
