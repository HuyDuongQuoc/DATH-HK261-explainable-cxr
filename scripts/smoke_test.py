import argparse
import json
from pathlib import Path

import torch
from torch import nn

from src.models.factory import build_model
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/smoke_test",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    findings = config["data"][
        "target_findings"
    ]

    image_size = config[
        "preprocessing"
    ]["image_size"]

    model_name = config["model"]["name"]

    output_dir = (
        Path(args.output_dir)
        / model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(
        config["seed"]
    )

    model = build_model(
        model_name=model_name,
        num_classes=len(findings),
        pretrained=False,
    )

    images = torch.randn(
        2,
        3,
        image_size,
        image_size,
    )

    labels = torch.randint(
        low=0,
        high=2,
        size=(2, len(findings)),
    ).float()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
    )

    logits = model(images)

    if logits.shape != labels.shape:
        raise ValueError(
            f"Unexpected logits shape: "
            f"{logits.shape}"
        )

    loss = nn.BCEWithLogitsLoss()(
        logits,
        labels,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    if not gradients:
        raise RuntimeError(
            "No gradients were created"
        )

    if not all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    ):
        raise RuntimeError(
            "Nonfinite gradients"
        )

    optimizer.step()

    checkpoint_path = (
        output_dir / "smoke_test.pt"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    restored_model = build_model(
        model_name=model_name,
        num_classes=len(findings),
        pretrained=False,
    )

    restored_model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()
    restored_model.eval()

    with torch.no_grad():
        original_output = model(images)
        restored_output = restored_model(
            images
        )

    if not torch.allclose(
        original_output,
        restored_output,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Checkpoint reload changed output"
        )

    result = {
        "status": "success",
        "model": model_name,
        "input_shape": list(
            images.shape
        ),
        "output_shape": list(
            logits.shape
        ),
        "loss": float(loss.item()),
        "backward_finite": True,
        "checkpoint_reload_valid": True,
        "data": "synthetic",
    }

    (
        output_dir / "result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(result)


if __name__ == "__main__":
    main()