import torch
import torch.nn.functional as F


ACTIVATION_TRANSFORMS = ("softplus", "relu")
MASK_TYPES = ("union", "majority", "soft")

# With 3 readers, soft masks take values k/3. A 0.5 threshold separates
# ">= 2/3 readers" from "1/3 reader" inside boxes (box borders get intermediate
# values from area resizing).
MAJORITY_THRESHOLD = 0.5


def positive_part(activation, transform="softplus"):
    """
    Turn a CAM into non-negative "energy".

    Transforms:
        softplus: default. Always has a gradient, but softplus(0) = ln 2, so the
            activation outside the box must be strongly negative to reach zero energy.
        relu: as in the Energy loss of Rao et al. (ICCV 2023); keeps the positive part only.
    """
    if transform == "softplus":
        return F.softplus(activation)
    if transform == "relu":
        return F.relu(activation)
    raise ValueError(f"Unknown activation transform: {transform}")


def select_target_masks(batch, mask_type="union"):
    """
    Build the target region for L_loc.

    Mask types:
        union: pixels inside a box from at least one reader (default).
        majority: pixels inside boxes from most readers (soft mask > 0.5).
        soft: fraction of readers whose box covers the pixel, used as a weight.

    Returns:
        (target_mask [B, C, H, W] in [0, 1], valid [B, C]). Pairs whose target mask
        is empty are marked invalid (e.g. majority is empty when boxes do not overlap).
    """
    valid = batch["mask_valid"]
    if mask_type == "union":
        target = (batch["union_masks"] > 0).float()
    elif mask_type == "majority":
        target = (batch["soft_masks"] > MAJORITY_THRESHOLD).float()
    elif mask_type == "soft":
        target = batch["soft_masks"].float()
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    nonempty = target.flatten(2).sum(-1) > 0
    return target, valid & nonempty


def localization_loss(
    activation,
    target_masks,
    mask_valid,
    eps: float = 1e-8,
    transform: str = "softplus",
):
    """
    L_loc = 1 - sum(A * G) / (sum(A) + eps), averaged over valid (image, class) pairs.

    A is the positive part of the CAM and G the target mask. G can be binary
    (union/majority) or a soft mask in [0, 1]; with a soft mask, energy in regions
    where few readers agree only counts partially.
    """
    A = positive_part(activation, transform)

    if A.shape[-2:] != target_masks.shape[-2:]:
        A = F.interpolate(
            A,
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    G = target_masks.to(A.dtype).clamp(0, 1)

    inside = (A * G).sum(dim=(2, 3))  # [B, C]
    total = A.sum(dim=(2, 3))  # [B, C]

    loss_bc = 1.0 - inside / (total + eps)

    if not mask_valid.any():
        return A.sum() * 0.0

    return loss_bc[mask_valid].mean()
