import numpy as np
import pandas as pd
import torch

from src.metrics.localization import (
    DEFAULT_IOU_THRESHOLDS,
    box_area_fraction,
    cam_iou,
    energy_inside_box,
    heatmap_correlation,
    iou_column,
    mean_iou_over_thresholds,
    pointing_game,
    resize_heatmap,
)
from src.xai.gradcam import GradCAM
from src.xai.occlusion import box_occlusion_drops, occlusion_maps
from src.xai.priors import pool_to_grid


ATTRIBUTION_METHODS = ("gradcam", "gradcam_penultimate", "occlusion")

# Các cột được tổng hợp theo finding nếu có trong records.
SUMMARY_METRICS = (
    ("pointing_hit_rate", "pointing_hit"),
    ("eib", "eib"),
    ("box_area", "box_area"),
    ("eib_over_area", "eib_over_area"),
    ("iou_mean", "iou_mean"),
    ("prior_eib", "prior_eib"),
    ("prior_hit", "prior_hit"),
    ("eib_minus_prior", "eib_minus_prior"),
    ("prior_corr", "prior_corr"),
    ("drop_inside", "drop_inside"),
    ("drop_outside", "drop_outside"),
)


class _Attributor:
    """Bọc các phương pháp attribution về cùng giao diện generate(images, class_idx)."""

    def __init__(self, model, model_name, method, occlusion_kwargs=None):
        if method not in ATTRIBUTION_METHODS:
            raise ValueError(f"Unknown attribution method: {method}")
        self.method = method
        self.model = model
        self.occlusion_kwargs = occlusion_kwargs or {}
        self.gradcam = None
        if method.startswith("gradcam"):
            layer = "penultimate" if method == "gradcam_penultimate" else "last"
            self.gradcam = GradCAM(model, model_name, target_layer=layer)

    def generate(self, images, class_idx):
        if self.gradcam is not None:
            cams, _ = self.gradcam.generate(images, class_idx=class_idx)
            return cams
        return occlusion_maps(self.model, images, class_idx, **self.occlusion_kwargs)

    def close(self):
        if self.gradcam is not None:
            self.gradcam.remove()


def collect_localization_records(
    model,
    model_name,
    loader,
    findings,
    device,
    thresholds=DEFAULT_IOU_THRESHOLDS,
    max_images=None,
    method="gradcam",
    priors=None,
    box_occlusion=False,
    occlusion_kwargs=None,
):
    """
    Với mỗi cặp (ảnh, class) có mask_valid: heatmap của đúng class có GT bbox
    (không phải class model dự đoán), rồi tính Pointing Game, EIB và IoU ở mọi
    threshold. Heatmap được resize về kích thước mask trước khi tính IoU/EIB.

    Cột bổ sung:
      box_area        tỉ lệ diện tích box (= mức ngẫu nhiên của EIB/Pointing)
      eib_over_area   EIB chia diện tích box (so được giữa split có box to/nhỏ khác nhau)
      iou_mean        IoU trung bình mọi ngưỡng (không cần chọn ngưỡng theo run)
      prior_*         nếu truyền priors [C, H, W]: EIB/Pointing của prior giải phẫu,
                      eib_minus_prior và tương quan heatmap–prior
      logit, drop_*   nếu box_occlusion=True: logit giảm khi che trong/ngoài box

    Loader phải trả union_masks và mask_valid (CXRDataset load_masks=True).
    """
    rows = []
    seen = 0
    attributor = _Attributor(model, model_name, method, occlusion_kwargs)

    try:
        for batch in loader:
            images = batch["image"].to(device)
            union_masks = batch["union_masks"]
            valid = batch["mask_valid"]

            for class_index, finding in enumerate(findings):
                selected = valid[:, class_index].nonzero(as_tuple=True)[0]
                if len(selected) == 0:
                    continue

                selected_images = images[selected.to(device)]
                cams = attributor.generate(selected_images, class_index)

                drops = None
                if box_occlusion:
                    drops = box_occlusion_drops(
                        model,
                        selected_images,
                        union_masks[selected, class_index],
                        class_index,
                    )

                prior = None if priors is None else priors[class_index]

                for position, sample_index in enumerate(selected.tolist()):
                    mask = union_masks[sample_index, class_index].numpy()
                    grid = cams[position]
                    heatmap = resize_heatmap(grid, mask.shape[0])
                    eib = energy_inside_box(heatmap, mask)
                    area = box_area_fraction(mask)

                    row = {
                        "image_id": batch["image_id"][sample_index],
                        "finding": finding,
                        "pointing_hit": pointing_game(grid, mask),
                        "eib": eib,
                        "box_area": area,
                        "eib_over_area": eib / area if area > 0 else np.nan,
                    }
                    for t in thresholds:
                        row[iou_column(t)] = cam_iou(heatmap, mask, t)
                    row["iou_mean"] = mean_iou_over_thresholds(row, thresholds)

                    if prior is not None:
                        prior_eib = energy_inside_box(prior, mask)
                        row["prior_eib"] = prior_eib
                        row["prior_hit"] = pointing_game(
                            pool_to_grid(prior, grid.shape), mask
                        )
                        row["eib_minus_prior"] = eib - prior_eib
                        row["prior_corr"] = heatmap_correlation(heatmap, prior)

                    if drops is not None:
                        for key, values in drops.items():
                            row[key] = float(values[position])

                    rows.append(row)

            seen += len(images)
            if max_images is not None and seen >= max_images:
                break
    finally:
        attributor.close()

    return pd.DataFrame(rows)


def summarize_by_finding(records, findings, threshold):
    """Bảng finding x {n, hit_rate, eib, iou, ...} tại threshold cố định, kèm dòng macro."""
    column = iou_column(threshold)
    metrics = [
        (name, source)
        for name, source in SUMMARY_METRICS
        if source in records.columns
    ]
    rows = []
    for finding in findings:
        subset = records[records["finding"] == finding]
        row = {"finding": finding, "n_pairs": len(subset)}
        for name, source in metrics:
            row[name] = subset[source].mean() if len(subset) else np.nan
        row["iou"] = subset[column].mean() if len(subset) else np.nan
        rows.append(row)

    table = pd.DataFrame(rows)
    macro = {"finding": "MACRO", "n_pairs": int(table["n_pairs"].sum())}
    for key in table.columns:
        if key not in ("finding", "n_pairs"):
            macro[key] = table[key].mean(skipna=True)
    return pd.concat([table, pd.DataFrame([macro])], ignore_index=True)
