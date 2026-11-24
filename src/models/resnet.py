import torch
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

    elif model_name == "vit_b_16_gap":
        features = model.feature_map(images)  # [B, D, 14, 14]
        head = model.head

    else:
        raise NotImplementedError(
            f"Spatial activation not supported for {model_name}"
        )

    # Tính CAM ở fp32 kể cả khi đang autocast fp16, để L_loc (tỉ lệ năng lượng)
    # không mất độ chính xác trước khi qua softplus/relu.
    with torch.autocast(device_type=features.device.type, enabled=False):
        activation = F.conv2d(
            features.float(),
            head.weight.float()[:, :, None, None],
            None if head.bias is None else head.bias.float(),
        )

    logits = activation.mean(dim=(2, 3))

    return logits, activation
