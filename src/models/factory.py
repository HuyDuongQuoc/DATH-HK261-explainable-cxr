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

SUPPORTED_MODELS = ("resnet50", "densenet121", "vit_b_16", "vit_b_16_gap")


def build_model(
    model_name,
    num_classes,
    pretrained=True,
    dilate_last_stage=False,
):
    """
    dilate_last_stage (chỉ ResNet-50): thay stride của layer4 bằng dilation,
    feature map cuối 14x14 thay vì 7x7 ở input 224, không thêm tham số nên vẫn
    dùng được trọng số ImageNet. Giúp CAM/L_loc mịn hơn cho tổn thương nhỏ.
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

    elif model_name == "vit_b_16":
        weights = (
            ViT_B_16_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        model = vit_b_16(weights=weights)

        model.heads.head = nn.Linear(
            model.heads.head.in_features,
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
    """Các tham số kiến trúc (ngoài tên/pretrained) đọc từ config["model"]."""
    return {
        "dilate_last_stage": bool(model_config.get("dilate_last_stage", False)),
    }


def classifier_head(model, model_name):
    """Lớp Linear cuối của từng kiến trúc (dùng cho khởi tạo bias theo prior)."""
    if model_name == "resnet50":
        return model.fc
    if model_name == "densenet121":
        return model.classifier
    if model_name == "vit_b_16":
        return model.heads.head
    if model_name == "vit_b_16_gap":
        return model.head
    raise ValueError(f"Unsupported model: {model_name}")
