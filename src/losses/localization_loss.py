import torch
import torch.nn.functional as F


def localization_loss(
    activation,
    union_masks,
    mask_valid,
    eps: float = 1e-8,
):

    A = F.softplus(activation)

    if A.shape[-2:] != union_masks.shape[-2:]:
        A = F.interpolate(
            A,
            size=union_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    G = (union_masks > 0).to(A.dtype)

    inside = (A * G).sum(dim=(2, 3))  # [B, C]
    total = A.sum(dim=(2, 3))  # [B, C]

    loss_bc = 1.0 - inside / (total + eps)

    if not mask_valid.any():
        return A.sum() * 0.0 

    return loss_bc[mask_valid].mean()
