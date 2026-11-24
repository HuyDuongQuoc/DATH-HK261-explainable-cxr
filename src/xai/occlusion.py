"""
Attribution và phép thử hành vi không dùng gradient, độc lập với CAM mà L_loc
tối ưu. Dùng để kiểm tra mô hình guided có thật sự dựa vào vùng tổn thương hay
chỉ "học cho đẹp" đúng heatmap được đánh giá.

Ảnh đầu vào đã chuẩn hóa ImageNet, nên fill=0 tương ứng màu xám trung bình.
"""

import numpy as np
import torch


@torch.no_grad()
def _class_logits(model, images, class_idx, chunk_size):
    outputs = []
    for start in range(0, len(images), chunk_size):
        outputs.append(model(images[start:start + chunk_size])[:, class_idx])
    return torch.cat(outputs).float()


def occlusion_positions(image_size, patch, stride):
    if patch <= 0 or stride <= 0 or patch > image_size:
        raise ValueError("Invalid occlusion patch/stride")
    starts = list(range(0, image_size - patch + 1, stride))
    if starts[-1] != image_size - patch:
        starts.append(image_size - patch)
    return starts


@torch.no_grad()
def occlusion_maps(
    model,
    images,
    class_idx,
    patch=32,
    stride=32,
    fill=0.0,
    chunk_size=64,
):
    """
    Occlusion sensitivity: che từng patch, đo logit class giảm bao nhiêu.

    Trả về np.ndarray [B, n, n] trong [0, 1] (ReLU của mức giảm, min-max theo
    ảnh), cùng quy ước với GradCAM.generate để dùng chung metric.
    """
    was_training = model.training
    model.eval()
    try:
        images = images.detach()
        batch, _, height, width = images.shape
        if height != width:
            raise ValueError("Occlusion expects square images")
        starts = occlusion_positions(height, patch, stride)
        n = len(starts)

        base = _class_logits(model, images, class_idx, chunk_size)  # [B]
        drops = torch.zeros(batch, n, n, device=images.device)

        for b in range(batch):
            variants = images[b:b + 1].repeat(n * n, 1, 1, 1)
            for k, (i, j) in enumerate(
                (i, j) for i in range(n) for j in range(n)
            ):
                y, x = starts[i], starts[j]
                variants[k, :, y:y + patch, x:x + patch] = fill
            scores = _class_logits(model, variants, class_idx, chunk_size)
            drops[b] = (base[b] - scores).reshape(n, n)
    finally:
        model.train(was_training)

    cam = drops.clamp(min=0).cpu().numpy()
    flat = cam.reshape(batch, -1)
    lo = flat.min(axis=1)[:, None, None]
    hi = flat.max(axis=1)[:, None, None]
    out = np.zeros_like(cam)
    ok = (hi > lo).reshape(-1)
    out[ok] = (cam[ok] - lo[ok]) / (hi[ok] - lo[ok])
    return out


@torch.no_grad()
def box_occlusion_drops(model, images, masks, class_idx, fill=0.0):
    """
    Phép thử hành vi: logit gốc, logit khi che toàn bộ vùng trong box và khi
    che toàn bộ vùng ngoài box.

    masks: [B, H, W] nhị phân cùng kích thước ảnh.
    Trả về dict các np.ndarray [B]: logit, drop_inside, drop_outside.
    Mô hình dựa vào tổn thương thật sẽ có drop_inside lớn và drop_outside nhỏ
    hơn baseline (so sánh paired giữa hai mô hình, không so trị tuyệt đối).
    """
    was_training = model.training
    model.eval()
    try:
        images = images.detach()
        g = (masks > 0).to(images.dtype)[:, None].to(images.device)
        hide_inside = images * (1 - g) + fill * g
        hide_outside = images * g + fill * (1 - g)
        base = model(images)[:, class_idx].float()
        inside = model(hide_inside)[:, class_idx].float()
        outside = model(hide_outside)[:, class_idx].float()
    finally:
        model.train(was_training)

    return {
        "logit": base.cpu().numpy(),
        "drop_inside": (base - inside).cpu().numpy(),
        "drop_outside": (base - outside).cpu().numpy(),
    }
