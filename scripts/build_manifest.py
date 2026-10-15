import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.annotations import (
    build_manifest,
    read_annotations,
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

    raw_root = Path(data_config["raw_root"])
    output_path = Path(
        data_config["manifest_path"]
    )
    metadata_path = Path(
        data_config["metadata_path"]
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. "
            "Use --overwrite only after changing "
            "the data protocol deliberately."
        )

    annotations = read_annotations(
        data_config["annotation_csv"]
    )

    findings = data_config["target_findings"]

    manifest, vote_table, _ = build_manifest(
        df=annotations,
        image_dir=raw_root / "train",
        findings=findings,
        expected_readers=data_config[
            "expected_readers"
        ],
        positive_votes=data_config[
            "positive_votes"
        ],
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        output_path,
        index=False,
    )

    vote_table.reset_index().to_csv(
        output_path.parent / "reader_votes.csv",
        index=False,
    )

    class_summary = []

    for finding in findings:
        positive = int(manifest[finding].sum())

        class_summary.append({
            "finding": finding,
            "positive_majority": positive,
            "negative_majority": (
                len(manifest) - positive
            ),
            "positive_rate": (
                positive / len(manifest)
            ),
        })

    pd.DataFrame(class_summary).to_csv(
        output_path.parent / "class_summary.csv",
        index=False,
    )

    metadata = {
        "target_findings": findings,
        "expected_readers": data_config[
            "expected_readers"
        ],
        "positive_votes": data_config[
            "positive_votes"
        ],
        "minority_policy": data_config[
            "minority_policy"
        ],
        "classification_rule": (
            "hard_majority_vote"
        ),
        "spatial_rule": (
            "reader_wise_union_then_soft_average"
        ),
        "mask_denominator": (
            "all_assigned_readers"
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Manifest created:", output_path)
    print("Images:", len(manifest))
    print(
        pd.DataFrame(class_summary).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()