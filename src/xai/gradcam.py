import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(
            self, 
            model, 
            model_name: str
        ):

        self.model = model
        self.model_name = model_name
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self.model.eval()

        self.target_layer = self._get_target_layer()

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)


    def generate(
            self, 
            input_tensor: torch.Tensor, 
            class_idx: int = None
        ):
    
            logits = self.model(input_tensor)
    
            if class_idx is None:
                class_idx = torch.argmax(logits, dim=1).item()
    
            self.model.zero_grad()
            score = logits[0, class_idx]
            score.backward(retain_graph=True)
    
            if self.model_name in ["resnet50", "densenet121"]:
                cam = self._process_cnn()
            
            elif self.model_name == "vit_b_16":
                cam = self._process_vit(
                    input_tensor.shape[2], 
                    input_tensor.shape[3]
                )
    
            return cam, class_idx


    def _get_target_layer(self):
        if self.model_name == "resnet50":
            return self.model.layer4[-1] 

        elif self.model_name == "densenet121":
            return self.model.features.norm5

        elif self.model_name == "vit_b_16":
            return self.model.encoder.ln

        else:
            raise ValueError(f"[GradCAM ERROR] target layer not valid for {self.model_name}")


    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach().clone()
            if output.requires_grad:
                output.register_hook(self._save_gradient)

        handle = self.target_layer.register_forward_hook(forward_hook)
        self.hook_handles.append(handle)


    def remove(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


    def _save_activation(self, module, input, output):
        self.activations = output.detach()


    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()


    def _process_cnn(self):
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        return self._normalize(cam.squeeze().cpu().numpy())


    def _process_vit(self, img_h: int, img_w: int):
        grads = self.gradients[:, 1:, :]  # (1, num_patches, D)
        acts = self.activations[:, 1:, :]  # (1, num_patches, D)

        weights = torch.mean(grads, dim=1, keepdim=True)  # (1, 1, D)
        cam = torch.sum(weights * acts, dim=-1)  # (1, num_patches)
        cam = F.relu(cam)

        grid_h = img_h // 16
        grid_w = img_w // 16
        cam = cam.reshape(grid_h, grid_w)

        return self._normalize(cam.cpu().numpy())


    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            return (cam - cam_min) / (cam_max - cam_min)
        return np.zeros_like(cam)


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