import numpy as np


def pointing_game(cam_grid, union_mask):
    # argmax trên GRID gốc (7×7/14×14), map cell-center → pixel, tránh sai số interpolate
    iy, ix = np.unravel_index(np.argmax(cam_grid), cam_grid.shape)
    y = (iy + 0.5) * union_mask.shape[0] / cam_grid.shape[0]
    x = (ix + 0.5) * union_mask.shape[1] / cam_grid.shape[1]
    return int(union_mask[int(y), int(x)] > 0)


def cam_iou(heatmap, union_mask, threshold, eps=1e-8):
    # heatmap phải đã được resize về cùng shape với union_mask (vd 224×224)
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
