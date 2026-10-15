import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from src.data.annotations import build_image_consensus


def resize_masks(
    soft_masks,
    union_masks,
    image_size,
):

    soft_tensor = torch.from_numpy(
        soft_masks.astype(np.float32)
    ).unsqueeze(0)

    union_tensor = torch.from_numpy(
        union_masks.astype(np.float32)
    ).unsqueeze(0)

    soft_resized = F.interpolate(
        soft_tensor,
        size=(image_size, image_size),
        mode="area",
    )[0]

    union_coverage = F.interpolate(
        union_tensor,
        size=(image_size, image_size),
        mode="area",
    )[0]

    union_resized = union_coverage.gt(0)

    return (
        soft_resized.clamp(0, 1).numpy(),
        union_resized.numpy(),
    )


def preprocess_image(
    image_path,
    annotations_for_image,
    findings,
    readers,
    image_size,
    positive_votes,
):
    with Image.open(image_path) as source:
        grayscale = source.convert("L")
        width, height = grayscale.size

        consensus = build_image_consensus(
            df_image=annotations_for_image,
            findings=findings,
            readers=readers,
            image_shape=(height, width),
            positive_votes=positive_votes,
        )

        resized_image = grayscale.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )

    soft_masks, union_masks = resize_masks(
        consensus["soft_masks"],
        consensus["union_masks"],
        image_size=image_size,
    )

    return {
        "image": resized_image,
        "soft_masks": soft_masks,
        "union_masks": union_masks,
        "labels": consensus["labels"],
        "mask_valid": consensus["mask_valid"],
        "positive_reader_counts": (consensus["positive_reader_counts"]),
        "num_readers": consensus["num_readers"],
    }