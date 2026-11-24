"""
Baseline "prior giải phẫu": heatmap cố định cho mỗi finding, bằng trung bình
union mask trên các cặp (ảnh, finding) dương tính của tập train. Heatmap này
không nhìn ảnh, nên một mô hình chỉ học "vị trí trung bình" của finding sẽ có
localization ngang prior. Localization của mô hình nên được báo cáo so với
prior, không chỉ so với 0.
"""

import os
from pathlib import Path

import cv2
import numpy as np


def accumulate_class_priors(loader, num_classes):
    """Trung bình union mask trên các cặp mask_valid, shape [C, H, W]."""
    total = None
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in loader:
        masks = (batch["union_masks"] > 0).float()
        valid = batch["mask_valid"]
        if total is None:
            total = np.zeros((num_classes, *masks.shape[-2:]), dtype=np.float64)
        weighted = (masks * valid[:, :, None, None].float()).sum(0)
        total += weighted.numpy()
        counts += valid.sum(0).numpy().astype(np.int64)
    if total is None:
        raise ValueError("Empty loader")
    priors = np.zeros_like(total, dtype=np.float32)
    nonzero = counts > 0
    priors[nonzero] = (total[nonzero] / counts[nonzero, None, None]).astype(np.float32)
    return priors, counts


def load_or_build_priors(cache_path, build_fn, findings):
    """Đọc prior từ cache .npz nếu khớp danh sách finding, nếu không thì build và lưu."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as stored:
            if stored["findings"].tolist() == list(findings):
                return stored["priors"], stored["counts"]
        raise ValueError(f"Prior cache {cache_path} was built for different findings")
    priors, counts = build_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Ghi ra file tạm rồi rename: nhiều container có thể build cùng lúc.
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        tmp_path,
        priors=priors,
        counts=counts,
        findings=np.asarray(findings),
    )
    os.replace(tmp_path, cache_path)
    return priors, counts


def pool_to_grid(prior, grid_shape):
    """Area-pool prior [H, W] về grid của heatmap (vd 7x7) để chấm Pointing Game cùng độ phân giải."""
    return cv2.resize(
        prior.astype(np.float32),
        (grid_shape[1], grid_shape[0]),
        interpolation=cv2.INTER_AREA,
    )
