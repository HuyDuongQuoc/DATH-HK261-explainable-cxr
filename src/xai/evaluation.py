import numpy as np
import pandas as pd
import torch

from src.metrics.localization import (
    DEFAULT_IOU_THRESHOLDS,
    cam_iou,
    energy_inside_box,
    iou_column,
    pointing_game,
    resize_heatmap,
)
from src.xai.gradcam import GradCAM


def collect_localization_records(
    model,
    model_name,
    loader,
    findings,
    device,
    thresholds=DEFAULT_IOU_THRESHOLDS,
    max_images=None,
):
    """
    Với mỗi cặp (ảnh, class) có mask_valid: Grad-CAM của đúng class có GT bbox
    (không phải class model dự đoán), rồi tính Pointing Game, EIB và IoU ở mọi
    threshold. Heatmap được resize về kích thước mask trước khi tính IoU/EIB.

    Loader phải trả union_masks và mask_valid (CXRDataset load_masks=True).
    """
    rows = []
    seen = 0

    with GradCAM(model, model_name) as gradcam:
        for batch in loader:
            images = batch["image"].to(device)
            union_masks = batch["union_masks"]
            valid = batch["mask_valid"]

            for class_index, finding in enumerate(findings):
                selected = valid[:, class_index].nonzero(as_tuple=True)[0]
                if len(selected) == 0:
                    continue

                cams, _ = gradcam.generate(
                    images[selected.to(device)],
                    class_idx=class_index,
                )

                for position, sample_index in enumerate(selected.tolist()):
                    mask = union_masks[sample_index, class_index].numpy()
                    grid = cams[position]
                    heatmap = resize_heatmap(grid, mask.shape[0])

                    row = {
                        "image_id": batch["image_id"][sample_index],
                        "finding": finding,
                        "pointing_hit": pointing_game(grid, mask),
                        "eib": energy_inside_box(heatmap, mask),
                    }
                    for t in thresholds:
                        row[iou_column(t)] = cam_iou(heatmap, mask, t)
                    rows.append(row)

            seen += len(images)
            if max_images is not None and seen >= max_images:
                break

    return pd.DataFrame(rows)


def summarize_by_finding(records, findings, threshold):
    """Bảng finding x {n, hit_rate, eib, iou} tại threshold cố định, kèm dòng macro."""
    column = iou_column(threshold)
    rows = []
    for finding in findings:
        subset = records[records["finding"] == finding]
        rows.append({
            "finding": finding,
            "n_pairs": len(subset),
            "pointing_hit_rate": subset["pointing_hit"].mean() if len(subset) else np.nan,
            "eib": subset["eib"].mean() if len(subset) else np.nan,
            "iou": subset[column].mean() if len(subset) else np.nan,
        })

    table = pd.DataFrame(rows)
    macro = {
        "finding": "MACRO",
        "n_pairs": int(table["n_pairs"].sum()),
        **{
            key: table[key].mean(skipna=True)
            for key in ("pointing_hit_rate", "eib", "iou")
        },
    }
    return pd.concat([table, pd.DataFrame([macro])], ignore_index=True)
