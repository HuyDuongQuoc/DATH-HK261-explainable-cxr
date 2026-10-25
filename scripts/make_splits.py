import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.splitting import (
    build_split_metadata,
    build_summary,
    create_split_tables,
    write_split_artifacts,
)
from src.utils.config import load_config


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

    manifest = pd.read_csv(manifest_path)
    findings = data_config["target_findings"]
    fractions = {
        "train": float(data_config["train_fraction"]),
        "val": float(data_config["val_fraction"]),
        "test": float(data_config["test_fraction"]),
    }
    split_tables, sizes = create_split_tables(
        manifest=manifest,
        findings=findings,
        fractions=fractions,
        seed=config["seed"],
    )
    summary = build_summary(split_tables, findings)
    split_metadata = build_split_metadata(
        manifest_path=manifest_path,
        split_tables=split_tables,
        findings=findings,
        fractions=fractions,
        sizes=sizes,
        seed=config["seed"],
    )
    write_split_artifacts(
        split_dir=split_dir,
        split_tables=split_tables,
        summary=summary,
        metadata=split_metadata,
        overwrite=args.overwrite,
    )

    print(summary.to_string(index=False))
    print(json.dumps(split_metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
