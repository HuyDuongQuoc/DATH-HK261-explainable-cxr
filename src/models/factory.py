from torch import nn

from torchvision.models import (
    DenseNet121_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    densenet121,
    resnet50,
    vit_b_16,
)


def build_model(
    model_name,
    num_classes,
    pretrained=True,
):
    if model_name == "resnet50":
        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )

        model = resnet50(weights=weights)

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

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model