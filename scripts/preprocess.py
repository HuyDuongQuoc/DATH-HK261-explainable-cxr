import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.annotations import (
    get_reader_roster,
    read_annotations,
)
from src.data.preprocessing import (
    preprocess_image,
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
    preprocessing_config = config["preprocessing"    ]

    raw_root = Path(data_config["raw_root"])
    processed_root = Path(
        data_config["processed_root"]
    )

    image_output_dir = processed_root / "images"

    mask_output_dir = processed_root / "masks"

    processed_manifest_path = processed_root / "manifest.csv"

    if (
        processed_manifest_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            "Processed manifest already exists. "
            "Use --overwrite only after changing "
            "the preprocessing protocol."
        )

    annotations = read_annotations(
        data_config["annotation_csv"]
    )

    manifest = pd.read_csv(
        data_config["manifest_path"]
    )

    findings = data_config["target_findings"]

    roster = get_reader_roster(
        annotations,
        expected_readers=data_config[
            "expected_readers"
        ],
    )

    annotation_groups = {
        image_id: group
        for image_id, group in annotations.groupby(
            "image_id"
        )
    }

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_rows = []

    for row in manifest.itertuples(index=False):
        image_id = row.image_id

        raw_image_path = raw_root / row.raw_image_path

        if image_id not in annotation_groups:
            raise ValueError(
                f"No annotations for {image_id}"
            )

        result = preprocess_image(
            image_path=raw_image_path,
            annotations_for_image=annotation_groups[image_id],
            findings=findings,
            readers=roster[image_id],
            image_size=preprocessing_config["image_size"],
            positive_votes=data_config["positive_votes"],
        )

        output_image_path = image_output_dir / f"{image_id}.png"

        output_mask_path = mask_output_dir / f"{image_id}.npz"

        result["image"].save(
            output_image_path
        )

        np.savez_compressed(
            output_mask_path,

            soft_masks=result[
                "soft_masks"
            ].astype(np.float16),

            union_masks=result[
                "union_masks"
            ].astype(np.uint8),

            labels=result[
                "labels"
            ].astype(np.uint8),

            mask_valid=result[
                "mask_valid"
            ].astype(bool),

            positive_reader_counts=result[
                "positive_reader_counts"
            ].astype(np.uint8),

            class_names=np.asarray(
                findings,
                dtype=np.str_,
            ),

            num_readers=np.asarray(
                result["num_readers"],
                dtype=np.uint8,
            ),
        )

        output_row = {
            "image_id": image_id,
            "image_path": (
                f"images/{image_id}.png"
            ),
            "mask_path": (
                f"masks/{image_id}.npz"
            ),
        }

        manifest_row = manifest[
            manifest["image_id"].eq(image_id)
        ].iloc[0]

        for class_index, finding in enumerate(findings):
            manifest_label = int(
                manifest_row[finding]
            )

            generated_label = int(
                result["labels"][class_index]
            )

            if manifest_label != generated_label:
                raise ValueError(
                    f"Label mismatch: {image_id}, "
                    f"{finding}"
                )

            output_row[finding] = manifest_label

        processed_rows.append(output_row)

        if len(processed_rows) % 500 == 0:
            print(
                f"Processed {len(processed_rows)} "
                f"/ {len(manifest)} images"
            )

    processed_manifest = pd.DataFrame(processed_rows)

    processed_manifest.to_csv(
        processed_manifest_path,
        index=False,
    )

    raw_split_dir = Path(
        data_config["split_dir"]
    )

    processed_split_dir = processed_root / "splits"

    processed_split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name in ["train", "val"]:
        raw_split = pd.read_csv(
            raw_split_dir / f"{split_name}.csv"
        )

        split_ids = set(
            raw_split["image_id"]
        )

        processed_split = (
            processed_manifest[
                processed_manifest[
                    "image_id"
                ].isin(split_ids)
            ]
            .sort_values("image_id")
            .copy()
        )

        if len(processed_split) != len(raw_split):
            raise ValueError(
                f"Processed {split_name} split "
                "does not match original split"
            )

        processed_split.to_csv(
            processed_split_dir
            / f"{split_name}.csv",
            index=False,
        )

    metadata = {
        "target_findings": findings,
        "image_size": preprocessing_config["image_size"],
        "classification_rule": ("hard_majority_vote" ),
        "positive_votes": data_config["positive_votes"],
        "minority_policy": data_config["minority_policy"],
        "mask_rule": ("reader_wise_union_then_soft_average"),
        "mask_denominator": ("all_assigned_readers"),
        "soft_mask_resize": (
            preprocessing_config[
                "soft_mask_resize_mode"
            ]
        ),
        "image_resize": (
            preprocessing_config[
                "image_resize_mode"
            ]
        ),
    }

    (processed_root / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Preprocessing completed:",
        processed_root,
    )


if __name__ == "__main__":
    main()