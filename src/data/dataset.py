from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import build_image_transform

class CXRDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        processed_root,
        findings,
        image_size,
        mean,
        std,
    ):
        self.table = pd.read_csv(manifest_path)
        self.root = Path(processed_root)
        self.findings = list(findings)
        self.image_size = image_size
        self.transform = build_image_transform(
            mean=mean,
            std=std,
        )

        if not self.table["image_id"].is_unique:
            raise ValueError(
                "Duplicate image IDs in manifest"
            )

        missing_columns = set(self.findings) - set(self.table.columns)

        if missing_columns:
            raise ValueError(
                f"Missing label columns: "
                f"{sorted(missing_columns)}"
            )

        label_values = self.table[
            self.findings
        ].to_numpy()

        if not np.isin(
            label_values,
            [0, 1],
        ).all():
            raise ValueError("Labels must be binary majority labels")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]

        image_path = self.root / row["image_path"]

        mask_path = self.root / row["mask_path"]

        with Image.open(image_path) as source:
            if source.size != (
                self.image_size,
                self.image_size,
            ):
                raise ValueError(
                    f"Unexpected image size: "
                    f"{image_path}, {source.size}"
                )

            image = self.transform(
                source.convert("L")
            )

        labels = torch.tensor(
            row[self.findings].to_numpy(
                dtype=np.float32
            ),
            dtype=torch.float32,
        )

        with np.load(
            mask_path,
            allow_pickle=False,
        ) as stored:
            class_names = (
                stored["class_names"].tolist()
            )

            soft_masks = torch.from_numpy(
                stored["soft_masks"].astype(
                    np.float32
                )
            )

            union_masks = torch.from_numpy(
                stored["union_masks"].astype(
                    np.float32
                )
            )

            stored_labels = torch.from_numpy(
                stored["labels"].astype(
                    np.float32
                )
            )

            mask_valid = torch.from_numpy(
                stored["mask_valid"].astype(
                    bool
                )
            )

            positive_reader_counts = (
                torch.from_numpy(
                    stored[
                        "positive_reader_counts"
                    ].astype(np.int64)
                )
            )

        if class_names != self.findings:
            raise ValueError(
                f"Class order mismatch: {mask_path}"
            )

        if not torch.equal(
            labels,
            stored_labels,
        ):
            raise ValueError(
                f"Manifest/mask label mismatch: "
                f"{row['image_id']}"
            )

        expected_mask_shape = (
            len(self.findings),
            self.image_size,
            self.image_size,
        )

        if tuple(soft_masks.shape) != expected_mask_shape:
            raise ValueError(
                "Unexpected soft-mask shape"
            )

        if tuple(union_masks.shape) != expected_mask_shape:
            raise ValueError("Unexpected union-mask shape")

        if not torch.isfinite(soft_masks).all():
            raise ValueError("Soft mask contains NaN/Inf")

        if soft_masks.min() < 0 or soft_masks.max() > 1:
            raise ValueError("Soft mask outside [0, 1]")

        expected_valid = (
            labels.eq(1)
            & soft_masks
            .flatten(1)
            .sum(1)
            .gt(0)
        )

        if not torch.equal(
            mask_valid,
            expected_valid,
        ):
            raise ValueError(
                "mask_valid does not match "
                "label/mask state"
            )

        return {
            "image": image,
            "labels": labels,
            "soft_masks": soft_masks,
            "union_masks": union_masks,
            "mask_valid": mask_valid,
            "positive_reader_counts": positive_reader_counts,
            "image_id": str(
                row["image_id"]
            ),
        }