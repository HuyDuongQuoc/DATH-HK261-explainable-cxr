import torch.nn.functional as F


def forward_with_activation(
    model,
    model_name,
    images,
):
    """
    Trả về (logits, activation) với activation [B, C, h, w] còn graph.

    CAM-style: áp trọng số của lớp phân loại lên feature map cuối
    (1x1 conv), nên activation.mean((2, 3)) == logits.
    Logits giống hệt model(images).
    """
    if model_name == "resnet50":
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        features = model.layer4(x)
        head = model.fc

    elif model_name == "densenet121":
        features = F.relu(model.features(images))
        head = model.classifier

    else:
        raise NotImplementedError(
            f"Spatial activation not supported for {model_name}"
        )

    activation = F.conv2d(
        features,
        head.weight[:, :, None, None],
        head.bias,
    )

    logits = activation.mean(dim=(2, 3))

    return logits, activation
