from torch import nn

from torchvision.models import (
    DenseNet121_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    densenet121,
    resnet50,
    vit_b_16,
)

from src.models.vit_gap import ViTGAP

SUPPORTED_MODELS = ("resnet50", "densenet121", "vit_b_16_gap")


def build_model(
    model_name,
    num_classes,
    pretrained=True,
    dilate_last_stage=False,
):
    """
    Build a classifier with a new linear head of size num_classes.

    Args:
        dilate_last_stage: ResNet-50 only. Replace the stride of layer4 with
            dilation, giving a 14x14 instead of 7x7 feature map at 224 px. No new
            parameters, so ImageNet weights still load. Gives finer CAMs for small lesions.
    """
    if dilate_last_stage and model_name != "resnet50":
        raise ValueError("dilate_last_stage is only supported for resnet50")

    if model_name == "resnet50":
        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )

        model = resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, False, bool(dilate_last_stage)],
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    elif model_name == "densenet121":
        weights = (
            DenseNet121_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        model = densenet121(
            weights=weights
        )

        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes,
        )

    elif model_name == "vit_b_16_gap":
        weights = (
            ViT_B_16_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        model = ViTGAP(vit_b_16(weights=weights), num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

def model_kwargs_from_config(model_config):
    """Architecture options (besides name/pretrained) read from config["model"]."""
    return {
        "dilate_last_stage": bool(model_config.get("dilate_last_stage", False)),
    }

