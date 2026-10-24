import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_SIZE = 1024

NO_FINDING = "No finding"

READER_ID = "consensus"


def load_class_ids(train_csv):
    train = pd.read_csv(train_csv)

    train = train.loc[
        :,
        ~train.columns.str.startswith("Unnamed"),
    ]

    mapping = (
        train.drop_duplicates("class_name")
        .set_index("class_name")["class_id"]
        .astype(int)
        .sort_values()
    )

    return mapping


def add_no_finding_rows(
    source,
    image_ids,
    dimensions,
):
    rows = []

    for image_id in image_ids:
        rows.append({
            "image_id": image_id,
            "class_name": NO_FINDING,
            "x_min": np.nan,
            "y_min": np.nan,
            "x_max": np.nan,
            "y_max": np.nan,
            "width": dimensions.loc[image_id, "width"],
            "height": dimensions.loc[image_id, "height"],
        })

    if rows:
        source = pd.concat(
            [source, pd.DataFrame(rows)],
            ignore_index=True,
        )

    return source


def scale_annotations(source):
    df = source.rename(
        columns={
            "x_min": "raw_x_min",
            "y_min": "raw_y_min",
            "x_max": "raw_x_max",
            "y_max": "raw_y_max",
            "width": "raw_width",
            "height": "raw_height",
        }
    )

    df["raw_width"] = df["raw_width"].astype(float)
    df["raw_height"] = df["raw_height"].astype(float)

    df["scale_x"] = IMAGE_SIZE / df["raw_width"]
    df["scale_y"] = IMAGE_SIZE / df["raw_height"]

    for axis in ["x", "y"]:
        scale = df[f"scale_{axis}"]

        df[f"{axis}_min"] = np.round(
            df[f"raw_{axis}_min"] * scale
        )

        df[f"{axis}_max"] = np.round(
            df[f"raw_{axis}_max"] * scale
        )

    return df


def validate(df, target_classes, image_dir):
    boxes = df.dropna(subset=["x_min"])

    expected = np.round(
        boxes["raw_x_min"] * boxes["scale_x"]
    )

    if not np.allclose(boxes["x_min"], expected):
        raise ValueError("x_min scaling mismatch")

    expected = np.round(
        boxes["raw_y_min"] * boxes["scale_y"]
    )

    if not np.allclose(boxes["y_min"], expected):
        raise ValueError("y_min scaling mismatch")

    if (boxes["x_min"] < 0).any() or (boxes["y_min"] < 0).any():
        raise ValueError("Negative scaled coordinates")

    if (boxes["x_max"] > IMAGE_SIZE).any() or (
        boxes["y_max"] > IMAGE_SIZE
    ).any():
        raise ValueError("Scaled coordinates exceed image size")

    if (boxes["x_max"] <= boxes["x_min"]).any():
        raise ValueError("Degenerate scaled boxes in x")

    if (boxes["y_max"] <= boxes["y_min"]).any():
        raise ValueError("Degenerate scaled boxes in y")

    unknown = set(df["class_name"]) - set(target_classes)
    if unknown:
        raise ValueError(f"Unknown classes: {sorted(unknown)}")

    images = set(df["image_id"])

    if image_dir is not None:
        image_dir = Path(image_dir)

        expected_ids = {
            path.stem
            for path in image_dir.glob("*.jpg")
        }

        if images != expected_ids:
            raise ValueError(
                "Test CSV images do not match image directory"
            )

    normal = df.loc[
        df["class_name"].eq(NO_FINDING),
        "image_id",
    ]

    abnormal = df.loc[
        ~df["class_name"].eq(NO_FINDING),
        "image_id",
    ]

    if not set(normal).isdisjoint(set(abnormal)):
        raise ValueError("Images marked both normal and abnormal")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="data/raw/test_source.csv",
    )

    parser.add_argument(
        "--train-csv",
        default="data/raw/train.csv",
    )

    parser.add_argument(
        "--output",
        default="data/raw/test.csv",
    )

    parser.add_argument(
        "--image-dir",
        default="data/raw/test",
    )

    args = parser.parse_args()

    class_ids = load_class_ids(args.train_csv)

    target_classes = list(class_ids.index)

    source = pd.read_csv(args.source)

    required = {
        "image_id",
        "class_name",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "width",
        "height",
    }

    missing = required - set(source.columns)

    if missing:
        raise ValueError(
            f"Missing source columns: {sorted(missing)}"
        )

    dimensions = source.drop_duplicates(
        "image_id"
    ).set_index("image_id")[["width", "height"]]

    dropped = source.loc[
        ~source["class_name"].isin(target_classes),
    ]

    source = source.loc[
        source["class_name"].isin(target_classes),
    ].copy()

    empty_images = sorted(
        set(dropped["image_id"]) - set(source["image_id"])
    )

    source = add_no_finding_rows(
        source,
        empty_images,
        dimensions,
    )

    df = scale_annotations(source)

    df["class_id"] = df["class_name"].map(class_ids)

    df["rad_id"] = READER_ID

    df = df.sort_values(
        ["image_id", "class_id", "x_min", "y_min"],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    df = df[
        [
            "image_id",
            "class_name",
            "class_id",
            "rad_id",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "raw_x_min",
            "raw_x_max",
            "raw_y_min",
            "raw_y_max",
            "raw_width",
            "raw_height",
            "scale_x",
            "scale_y",
        ]
    ]

    validate(df, target_classes, args.image_dir)

    df.index.name = "Unnamed: 0"

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(output)

    print("Test annotations created:", output)
    print("Rows:", len(df))
    print("Images:", df["image_id"].nunique())
    print(
        "Dropped non-target rows:",
        len(dropped),
        "from images:",
        len(set(dropped["image_id"])),
    )
    print(
        "Images with only non-target findings "
        f"(set to {NO_FINDING}):",
        len(empty_images),
        empty_images,
    )
    print(
        "Normal images:",
        df.loc[
            df["class_name"].eq(NO_FINDING),
            "image_id",
        ].nunique(),
    )


if __name__ == "__main__":
    main()
