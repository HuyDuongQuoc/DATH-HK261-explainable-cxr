import torch
import torch.nn.functional as F


ACTIVATION_TRANSFORMS = ("softplus", "relu")
MASK_TYPES = ("union", "majority", "soft")

# Với 3 reader, soft mask nhận giá trị k/3. Ngưỡng 0.5 tách đúng
# "≥2/3 reader" khỏi "1/3 reader" ở phần trong box (biên có giá trị trung gian
# do resize kiểu area).
MAJORITY_THRESHOLD = 0.5


def positive_part(activation, transform="softplus"):
    """
    Biến CAM thành "năng lượng" không âm.

    - softplus (mặc định, giữ hành vi cũ): luôn có gradient, nhưng softplus(0)=ln2
      nên để năng lượng ngoài box về 0, activation ngoài box phải rất âm.
    - relu: giống Energy loss của Rao et al. (ICCV 2023), chỉ tính phần dương.
    """
    if transform == "softplus":
        return F.softplus(activation)
    if transform == "relu":
        return F.relu(activation)
    raise ValueError(f"Unknown activation transform: {transform}")


def select_target_masks(batch, mask_type="union"):
    """
    Trả về (target_mask [B,C,H,W] trong [0,1], valid [B,C]) theo loại mask.

    - union:    vùng có ít nhất 1 reader vẽ box (hành vi cũ).
    - majority: vùng có đa số reader vẽ box (soft > 0.5).
    - soft:     tỉ lệ reader vẽ box, dùng làm trọng số liên tục.

    valid bị thu hẹp thêm nếu mask đích rỗng (ví dụ majority rỗng khi các reader
    vẽ box không chồng nhau).
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
    L_loc = 1 - sum(A * G) / (sum(A) + eps), trung bình trên các cặp
    (ảnh, class) hợp lệ.

    target_masks có thể là mask nhị phân (union/majority) hoặc soft mask trong
    [0, 1]; với soft mask, năng lượng trong vùng ít reader đồng ý chỉ được tính
    một phần.
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
