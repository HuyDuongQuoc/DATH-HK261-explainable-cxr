"""
ViT-B/16 với head GAP trên patch token (ViT-GAP).

ViT gốc của torchvision phân loại bằng CLS token, nên không có feature map
không gian gắn với class. ViT-GAP giữ nguyên backbone (kể cả CLS token, để dùng
được position embedding 197 vị trí của trọng số ImageNet) nhưng bỏ đầu ra của
CLS: head tuyến tính áp lên từng patch token rồi lấy trung bình,

    activation[b, c, i, j] = W_c · token[b, i, j] + b_c     (grid 14x14)
    logits[b, c]           = mean_{i,j} activation[b, c, i, j]

tức tương đương GAP + linear của CNN. Vì vậy CAM, L_loc và Grad-CAM dùng lại
được y như ResNet/DenseNet.

Lưu ý: đây là kiến trúc khác ViT-CLS (``vit_b_16``), nên ablation L_loc phải so
với baseline ViT-GAP (λ=0), không so với ViT-CLS.
"""

import math

import torch
from torch import nn


class ViTGAP(nn.Module):
    def __init__(self, vit, num_classes):
        super().__init__()
        hidden_dim = vit.hidden_dim
        vit.heads = nn.Identity()  # head CLS không dùng tới
        self.backbone = vit
        self.head = nn.Linear(hidden_dim, num_classes)

    def encode(self, images, capture_layer=None, capture_fn=None):
        """
        Chạy encoder như torchvision, trả về token sau LayerNorm cuối [B, 1+N, D].
        Nếu capture_layer (chỉ số block) được đặt, gọi capture_fn(tokens) với
        output của block đó (dùng cho Grad-CAM lớp áp chót).
        """
        vit = self.backbone
        x = vit._process_input(images)
        batch_class_token = vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        encoder = vit.encoder
        x = encoder.dropout(x + encoder.pos_embedding)
        for index, layer in enumerate(encoder.layers):
            x = layer(x)
            if capture_fn is not None and index == capture_layer:
                capture_fn(x)
        return encoder.ln(x)

    @staticmethod
    def tokens_to_grid(tokens):
        """Patch token [B, N, D] -> feature map [B, D, s, s]."""
        batch, count, dim = tokens.shape
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError(f"Patch count {count} is not a square grid")
        return tokens.transpose(1, 2).reshape(batch, dim, side, side)

    def feature_map(self, images):
        """Feature map cuối (bỏ CLS) dạng [B, D, 14, 14]."""
        return self.tokens_to_grid(self.encode(images)[:, 1:])

    def forward(self, images):
        tokens = self.encode(images)[:, 1:]
        return self.head(tokens.mean(dim=1))
