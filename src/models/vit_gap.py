"""
ViT-B/16 with a GAP head over patch tokens (ViT-GAP).

The torchvision ViT classifies from the CLS token, so it has no class-specific
spatial feature map. ViT-GAP keeps the backbone (including the CLS token, so the
197 ImageNet position embeddings still fit) but ignores the CLS output: a linear
head is applied to every patch token and the results are averaged,

    activation[b, c, i, j] = W_c · token[b, i, j] + b_c     (grid 14x14)
    logits[b, c]           = mean_{i,j} activation[b, c, i, j]

which matches GAP + linear in a CNN. CAM, L_loc and Grad-CAM therefore work
exactly as for ResNet/DenseNet.

Note: the L_loc ablation for this model compares against a ViT-GAP baseline (λ=0).
"""

import math

import torch
from torch import nn


class ViTGAP(nn.Module):
    def __init__(self, vit, num_classes):
        super().__init__()
        hidden_dim = vit.hidden_dim
        vit.heads = nn.Identity()  # the CLS head is not used
        self.backbone = vit
        self.head = nn.Linear(hidden_dim, num_classes)

    def encode(self, images, capture_layer=None, capture_fn=None):
        """
        Run the encoder as torchvision does and return tokens after the final
        LayerNorm, [B, 1 + N, D].

        Args:
            capture_layer: optional block index; capture_fn(tokens) is called on the
                output of that block (used for penultimate-layer Grad-CAM).
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
        """Last feature map without the CLS token, [B, D, 14, 14]."""
        return self.tokens_to_grid(self.encode(images)[:, 1:])

    def forward(self, images):
        tokens = self.encode(images)[:, 1:]
        return self.head(tokens.mean(dim=1))
