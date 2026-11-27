"""
Gradient-free attribution and behavioural tests, independent of the CAM that
L_loc optimizes. They check whether a guided model really relies on the lesion
or only learns to produce the heatmap that is being evaluated.

Inputs are ImageNet-normalized, so fill=0 corresponds to the mean gray value.
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
    Occlusion sensitivity: hide one patch at a time and measure the class logit drop.

    Returns:
        np.ndarray [B, n, n] in [0, 1] (ReLU of the drop, min-max scaled per image),
        the same convention as GradCAM.generate so the same metrics apply.
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
    Behavioural test: the original logit, and its drop when hiding everything
    inside the box or everything outside the box.

    Args:
        masks: binary [B, H, W] masks with the image size.

    Returns:
        dict of np.ndarray [B]: logit, drop_inside, drop_outside. A model that
        relies on the lesion has a larger drop_inside than its baseline (compare
        paired models, not absolute values).
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
