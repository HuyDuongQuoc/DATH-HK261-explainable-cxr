import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SUPPORTED_MODELS = ("resnet50", "densenet121", "vit_b_16", "vit_b_16_gap")
TARGET_LAYERS = ("last", "penultimate")


class GradCAM:
    """
    Grad-CAM cho ResNet50, DenseNet121 và ViT-B/16.

    - CNN: tự forward tới feature map cuối rồi gắn hook gradient lên chính tensor
      đó. Không dùng module backward hook nên không vướng ReLU inplace của DenseNet.
    - ViT: hook vào ln_1 của block cuối. Head chỉ đọc CLS token, nên gradient tại
      output encoder (hay output block cuối) của patch token luôn bằng 0; tại ln_1
      patch token vẫn ảnh hưởng CLS qua attention nên gradient khác 0.

    generate() trả về (cam, class_idx):
        cam:       np.ndarray [B, h, w] trong [0, 1], h x w là grid gốc (7x7 / 14x14)
        class_idx: list[int] độ dài B
    """

    def __init__(self, model, model_name: str, target_layer: str = "last"):
        """
        target_layer (chỉ CNN):
          - "last": feature map cuối (mặc định). Với GAP + linear, Grad-CAM ở đây
            trùng CAM mà L_loc tối ưu, nên không độc lập với loss.
          - "penultimate": output của layer3 (ResNet) / denseblock3 (DenseNet) /
            block áp chót (ViT-GAP), grid 14x14, không phải đại lượng được tối ưu
            trực tiếp.
        ViT-GAP ở lớp cuối được xử lý như CNN: Grad-CAM trùng CAM trên patch token.
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"[GradCAM ERROR] unsupported model: {model_name}"
            )
        if target_layer not in TARGET_LAYERS:
            raise ValueError(f"[GradCAM ERROR] unsupported target_layer: {target_layer}")
        if target_layer != "last" and model_name == "vit_b_16":
            raise ValueError("[GradCAM ERROR] target_layer is only configurable for CNNs")

        self.model = model
        self.model_name = model_name
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []

        if model_name == "vit_b_16":
            target = self.model.encoder.layers[-1].ln_1
            self.hook_handles.append(
                target.register_forward_hook(self._save_vit_activation)
            )

    # ---------- hooks ----------
    def _save_vit_activation(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def remove(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    # ---------- forward ----------
    def _forward(self, x):
        m = self.model

        penultimate = self.target_layer == "penultimate"

        if self.model_name == "resnet50":
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            x = m.layer3(m.layer2(m.layer1(x)))
            if penultimate:
                self._capture(x)
            features = m.layer4(x)
            head = m.fc

        elif self.model_name == "vit_b_16_gap":
            capture_layer = len(m.backbone.encoder.layers) - 2
            tokens = m.encode(
                x,
                capture_layer=capture_layer if penultimate else None,
                capture_fn=self._capture if penultimate else None,
            )
            features = m.tokens_to_grid(tokens[:, 1:])
            head = m.head

        elif self.model_name == "densenet121":
            for name, module in m.features.named_children():
                x = module(x)
                if penultimate and name == "denseblock3":
                    self._capture(x)
            features = F.relu(x)  # không inplace
            head = m.classifier

        else:
            return m(x)

        if not penultimate:
            self._capture(features)

        return head(torch.flatten(F.adaptive_avg_pool2d(features, 1), 1))

    def _capture(self, tensor):
        tensor.register_hook(self._save_gradient)
        self.activations = tensor

    def generate(self, input_tensor: torch.Tensor, class_idx=None):
        """class_idx: None (argmax), int, hoặc list/tensor độ dài B."""
        was_training = self.model.training
        self.model.eval()
        self.activations = None
        self.gradients = None

        try:
            with torch.enable_grad():
                # requires_grad để graph luôn tồn tại kể cả khi model bị freeze
                x = input_tensor.detach().clone().requires_grad_(True)
                logits = self._forward(x)

                batch = logits.shape[0]
                if class_idx is None:
                    idx = logits.argmax(dim=1)
                elif isinstance(class_idx, int):
                    idx = torch.full((batch,), class_idx, dtype=torch.long)
                else:
                    idx = torch.as_tensor(class_idx, dtype=torch.long)
                if idx.shape != (batch,):
                    raise ValueError("class_idx must be an int or length-B sequence")
                idx = idx.to(logits.device)

                self.model.zero_grad(set_to_none=True)
                score = logits.gather(1, idx[:, None]).sum()
                score.backward()

            if self.gradients is None:
                raise RuntimeError("[GradCAM ERROR] no gradient captured")

            if self.model_name == "vit_b_16" or (
                self.model_name == "vit_b_16_gap" and self.target_layer == "penultimate"
            ):
                cam = self._process_vit()  # activation là token [B, 1+N, D]
            else:
                cam = self._process_cnn()
        finally:
            self.model.train(was_training)

        return cam, idx.tolist()

    # ---------- cam ----------
    def _process_cnn(self):
        grads = self.gradients.detach()
        acts = self.activations.detach()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1))  # [B, h, w]

        return self._normalize(cam.cpu().numpy())

    def _process_vit(self):
        grads = self.gradients.detach()[:, 1:, :]  # bỏ CLS
        acts = self.activations.detach()[:, 1:, :]

        weights = grads.mean(dim=1, keepdim=True)
        cam = F.relu((weights * acts).sum(dim=-1))  # [B, N]

        side = math.isqrt(cam.shape[1])
        if side * side != cam.shape[1]:
            raise ValueError(f"Patch count {cam.shape[1]} is not a square grid")

        return self._normalize(cam.reshape(-1, side, side).cpu().numpy())

    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        """Min-max theo từng ảnh; ảnh phẳng -> toàn 0."""
        flat = cam.reshape(cam.shape[0], -1)
        lo = flat.min(axis=1)[:, None, None]
        hi = flat.max(axis=1)[:, None, None]
        out = np.zeros_like(cam)
        ok = (hi > lo).reshape(-1)
        out[ok] = (cam[ok] - lo[ok]) / (hi[ok] - lo[ok])
        return out


# display
def overlay_heatmap(
    img_pil: Image.Image, 
    cam: np.ndarray, 
    alpha: float = 0.5
) -> np.ndarray:
    cam_resized = cv2.resize(
        cam, 
        (img_pil.width, img_pil.height)
    )
    
    heatmap = np.uint8(255 * cam_resized)
    
    heatmap_colored = cv2.applyColorMap(
        heatmap, 
        cv2.COLORMAP_JET
    )
    
    heatmap_colored = cv2.cvtColor(
        heatmap_colored, 
        cv2.COLOR_BGR2RGB
    )

    img_np = np.array(img_pil)

    blended = np.float32(heatmap_colored) * alpha + np.float32(img_np) * (1.0 - alpha)
    
    return np.uint8(blended)