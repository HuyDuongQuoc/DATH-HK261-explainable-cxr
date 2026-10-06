from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


ID_COLUMNS = [
    "image_id",
    "class_name",
    "rad_id",
]

BOX_COLUMNS = [
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]

NORMAL_CLASS = "No finding"


def read_annotations(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    df = df.loc[
        :,
        ~df.columns.str.startswith("Unnamed:")
    ].copy()

    required_columns = set(ID_COLUMNS + BOX_COLUMNS)
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    for column in ID_COLUMNS:
        if df[column].isna().any():
            raise ValueError(
                f"Missing values in column: {column}"
            )

        df[column] = df[column].astype(str).str.strip()

    for column in BOX_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def get_reader_roster(df, expected_readers=3):
    roster = (
        df.groupby("image_id")["rad_id"]
        .apply(lambda values: sorted(set(values)))
        .to_dict()
    )

    invalid = {
        image_id: readers
        for image_id, readers in roster.items()
        if len(readers) != expected_readers
    }

    if invalid:
        examples = list(invalid.items())[:5]

        raise ValueError(
            "Unexpected reader count. "
            f"Examples: {examples}"
        )

    return roster


def make_majority_labels(
    df,
    findings,
    expected_readers=3,
    positive_votes=2,
):
    # Create classification label by majority vote.
    findings = list(findings)

    if not findings:
        raise ValueError("findings is empty")

    available = set(df["class_name"]) - {NORMAL_CLASS}
    unknown = set(findings) - available

    if unknown:
        raise ValueError(
            f"Unknown findings: {sorted(unknown)}"
        )

    roster = get_reader_roster(
        df,
        expected_readers=expected_readers,
    )

    selected = df[
        df["class_name"].isin(findings)
    ].copy()

    vote_table = (
        selected.groupby(
            ["image_id", "class_name"]
        )["rad_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(
            index=sorted(roster),
            columns=findings,
            fill_value=0,
        )
    )

    labels = (
        vote_table >= positive_votes
    ).astype(np.uint8)

    labels.index.name = "image_id"

    return labels.reset_index(), vote_table, roster


def validate_bbox(row, width, height):
    coordinates = np.asarray(
        [
            row.x_min,
            row.y_min,
            row.x_max,
            row.y_max,
        ],
        dtype=float,
    )

    if not np.isfinite(coordinates).all():
        raise ValueError(
            f"Nonfinite bbox: {row.image_id}, "
            f"{row.class_name}, {row.rad_id}"
        )

    x0, y0, x1, y1 = coordinates

    if not (
        0 <= x0 < x1 <= width
        and 0 <= y0 < y1 <= height
    ):
        raise ValueError(
            f"BBox outside JPG dimensions: "
            f"image_id={row.image_id}, "
            f"bbox={coordinates.tolist()}, "
            f"image_size={(width, height)}"
        )

    return x0, y0, x1, y1


def build_image_consensus(
    df_image,
    findings,
    readers,
    image_shape,
    positive_votes=2,
):
    """
    Create majority labels and soft-pixel consensus masks for a picture.
    Process:
    1. Union bbox individually by doctors
    2. Add reader masks
    3. Divide by the total number of doctor.
    """
    height, width = image_shape
    findings = list(findings)

    if df_image["image_id"].nunique() != 1:
        raise ValueError(
            "df_image must contain one image only"
        )

    class_count = len(findings)

    vote_counts = np.zeros(
        (class_count, height, width),
        dtype=np.float32,
    )

    positive_reader_counts = np.zeros(
        class_count,
        dtype=np.int64,
    )

    for class_index, finding in enumerate(findings):
        for reader in readers:
            reader_rows = df_image[
                df_image["rad_id"].eq(reader)
                & df_image["class_name"].eq(finding)
            ]

            reader_mask = np.zeros(
                (height, width),
                dtype=np.float32,
            )

            if not reader_rows.empty:
                positive_reader_counts[class_index] += 1

            for row in reader_rows.itertuples(index=False):
                x0, y0, x1, y1 = validate_bbox(
                    row,
                    width=width,
                    height=height,
                )

                x0 = int(np.floor(x0))
                y0 = int(np.floor(y0))
                x1 = int(np.ceil(x1))
                y1 = int(np.ceil(y1))

                reader_mask[y0:y1, x0:x1] = 1.0

            vote_counts[class_index] += reader_mask

    num_readers = len(readers)

    soft_masks = vote_counts / float(num_readers)

    union_masks = (
        vote_counts > 0
    ).astype(np.uint8)

    labels = (
        positive_reader_counts >= positive_votes
    ).astype(np.uint8)

    mask_valid = (
        labels.astype(bool)
        & (
            union_masks
            .reshape(class_count, -1)
            .sum(axis=1)
            > 0
        )
    )

    return {
        "soft_masks": soft_masks,
        "union_masks": union_masks,
        "labels": labels,
        "mask_valid": mask_valid,
        "positive_reader_counts": positive_reader_counts,
        "num_readers": num_readers,
    }


def build_manifest(
    df,
    image_dir,
    findings,
    expected_readers=3,
    positive_votes=2,
):
    image_dir = Path(image_dir)

    labels, vote_table, roster = make_majority_labels(
        df=df,
        findings=findings,
        expected_readers=expected_readers,
        positive_votes=positive_votes,
    )

    rows = []

    for _, label_row in labels.iterrows():
        image_id = label_row["image_id"]
        image_path = image_dir / f"{image_id}.jpg"

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as image:
            width, height = image.size

        row = {
            "image_id": image_id,
            "raw_image_path": f"train/{image_id}.jpg",
            "raw_width": width,
            "raw_height": height,
        }

        for finding in findings:
            row[finding] = int(label_row[finding])

        rows.append(row)

    manifest = pd.DataFrame(rows)

    return manifest, vote_table, roster