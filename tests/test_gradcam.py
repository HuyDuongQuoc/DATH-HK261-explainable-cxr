import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.factory import build_model
from src.models.resnet import forward_with_activation
from src.xai.gradcam import GradCAM, overlay_heatmap


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(0)
    return torch.randn(3, 3, 224, 224)


@pytest.mark.parametrize(
    "name,grid", [("resnet50", 7), ("densenet121", 7), ("vit_b_16_gap", 14)]
)
def test_shape_range_batch_consistency_and_cleanup(name, grid, images):
    model = build_model(name, 14, pretrained=False)
    model.train()
    with GradCAM(model, name) as gradcam:
        cam, idx = gradcam.generate(images, [3, 5, 0])
        single, _ = gradcam.generate(images[:1], 3)
    assert cam.shape == (3, grid, grid)
    assert idx == [3, 5, 0]
    assert cam.min() >= 0 and cam.max() <= 1
    assert float(cam.max()) == 1.0
    assert np.allclose(cam[0], single[0], atol=1e-5)
    assert gradcam.hook_handles == []
    assert model.training  # train mode is restored


@pytest.mark.parametrize("name", ["resnet50", "densenet121"])
def test_cnn_gradcam_equals_cam_from_activation(name, images):
    # With GAP + linear, Grad-CAM equals the CAM used in L_loc (up to the bias)
    model = build_model(name, 14, pretrained=False).eval()
    head = model.fc if name == "resnet50" else model.classifier
    with GradCAM(model, name) as gradcam:
        cam, _ = gradcam.generate(images, 3)
    with torch.no_grad():
        _, activation = forward_with_activation(model, name, images)
        ref = F.relu(activation[:, 3] - head.bias[3]).numpy()
    for i in range(len(images)):
        r = (ref[i] - ref[i].min()) / (ref[i].max() - ref[i].min() + 1e-12)
        assert np.allclose(r, cam[i], atol=1e-4)


def test_default_class_is_argmax_and_validation(images):
    model = build_model("resnet50", 14, pretrained=False).eval()
    with torch.no_grad():
        expected = model(images[:2]).argmax(dim=1).tolist()
    with GradCAM(model, "resnet50") as gradcam:
        _, idx = gradcam.generate(images[:2])
        assert idx == expected
        with pytest.raises(ValueError):
            gradcam.generate(images[:2], [1, 2, 3])
    with pytest.raises(ValueError):
        GradCAM(model, "unknown_model")


def test_overlay_heatmap_shape():
    from PIL import Image
    img = Image.fromarray(np.zeros((64, 48, 3), dtype=np.uint8))
    out = overlay_heatmap(img, np.random.rand(7, 7).astype(np.float32))
    assert out.shape == (64, 48, 3) and out.dtype == np.uint8
