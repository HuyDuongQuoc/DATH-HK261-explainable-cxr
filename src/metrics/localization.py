import cv2
import numpy as np


DEFAULT_IOU_THRESHOLDS = tuple(np.round(np.arange(0.10, 0.91, 0.05), 2))


def resize_heatmap(cam_grid, size):
    cam = np.asarray(cam_grid, dtype=np.float32)
    return cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR)


def pointing_game(cam_grid, union_mask):
    iy, ix = np.unravel_index(np.argmax(cam_grid), cam_grid.shape)
    y = (iy + 0.5) * union_mask.shape[0] / cam_grid.shape[0]
    x = (ix + 0.5) * union_mask.shape[1] / cam_grid.shape[1]
    return int(union_mask[int(y), int(x)] > 0)


def cam_iou(heatmap, union_mask, threshold, eps=1e-8):
    if heatmap.shape != union_mask.shape:
        raise ValueError(
            f"heatmap {heatmap.shape} != union_mask {union_mask.shape}"
        )

    pred = heatmap >= threshold
    gt = union_mask > 0

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    return float(inter / (union + eps))


def energy_inside_box(heatmap, union_mask, eps=1e-8):
    if heatmap.shape != union_mask.shape:
        raise ValueError(
            f"heatmap {heatmap.shape} != union_mask {union_mask.shape}"
        )

    return float(
        (heatmap * (union_mask > 0)).sum() / (heatmap.sum() + eps)
    )


def iou_column(threshold):
    return f"iou@{float(threshold):.2f}"


def select_iou_threshold(records, thresholds=DEFAULT_IOU_THRESHOLDS):
    if len(records) == 0:
        raise ValueError("Cannot select IoU threshold from empty records")

    means = {
        float(t): float(records[iou_column(t)].mean())
        for t in thresholds
    }
    best = max(sorted(means), key=lambda t: means[t])
    return best, means


def box_area_fraction(union_mask):
    """Tỉ lệ diện tích ảnh nằm trong box = EIB/Pointing kỳ vọng của heatmap đều."""
    return float((np.asarray(union_mask) > 0).mean())


def mean_iou_over_thresholds(row, thresholds=DEFAULT_IOU_THRESHOLDS):
    """IoU trung bình trên mọi ngưỡng: không cần chọn ngưỡng riêng cho từng run."""
    return float(np.mean([row[iou_column(t)] for t in thresholds]))


def heatmap_correlation(heatmap, reference, eps=1e-8):
    """Pearson giữa heatmap và một heatmap tham chiếu (vd prior). Cao = ít phụ thuộc ảnh."""
    a = np.asarray(heatmap, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > eps else float("nan")
