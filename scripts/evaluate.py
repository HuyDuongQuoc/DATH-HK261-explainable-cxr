import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CXRDataset
from src.data.loader import build_loader_kwargs
from src.data.protocol import (
    load_split_metadata,
    validate_checkpoint_protocol,
)
from src.data.splitting import file_sha256
from src.metrics.classification import compute_classification_metrics
from src.models.factory import build_model
from src.training.engine import run_epoch
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.seed import set_seed


def build_predictions_table(image_ids, targets, probabilities, findings, threshold):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(np.uint8)
    table = {"image_id": list(image_ids)}

    for index, finding in enumerate(findings):
        table[f"target::{finding}"] = targets[:, index].astype(np.uint8)
        table[f"probability::{finding}"] = probabilities[:, index]
        table[f"prediction::{finding}"] = predictions[:, index]

    return pd.DataFrame(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    config = load_config(args.config)
    set_seed(config["seed"])
    findings = config["data"]["target_findings"]
    processed_root = Path(config["data"]["processed_root"])
    split_root = processed_root / "splits"
    split_metadata = load_split_metadata(split_root)
    assignment = split_metadata["assignment_sha256"]

    preprocessing = config["preprocessing"]
    dataset = CXRDataset(
        manifest_path=split_root / f"{args.split}.csv",
        processed_root=processed_root,
        findings=findings,
        image_size=preprocessing["image_size"],
        mean=preprocessing["mean"],
        std=preprocessing["std"],
        load_masks=False,
    )
    loader_kwargs = build_loader_kwargs(
        config["training"],
        pin_memory=torch.cuda.is_available(),
    )
    loader = DataLoader(
        dataset,
        shuffle=False,
        **loader_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(
        config["training"]["use_amp"] and device.type == "cuda"
    )
    model = build_model(
        model_name=config["model"]["name"],
        num_classes=len(findings),
        pretrained=False,
    ).to(device)
    checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        optimizer=None,
        device=device,
    )
    validate_checkpoint_protocol(
        checkpoint=checkpoint,
        model_name=config["model"]["name"],
        findings=findings,
        assignment_sha256=assignment,
    )
    if "pos_weight" not in checkpoint:
        raise ValueError("Checkpoint is missing pos_weight")

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=checkpoint["pos_weight"].to(device)
    )
    result = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        use_amp=amp_enabled,
        show_progress=config["training"]["show_progress"],
        description=f"Evaluate {args.split}",
    )
    metrics = compute_classification_metrics(
        targets=result["targets"],
        probabilities=result["probabilities"],
        findings=findings,
        threshold=args.threshold,
    )
    metrics.update(
        {
            "split": args.split,
            "threshold": args.threshold,
            "loss": float(result["loss"]),
            "num_images": len(dataset),
            "elapsed_seconds": result["elapsed_seconds"],
            "images_per_second": result["images_per_second"],
            "peak_gpu_memory_gb": result["peak_gpu_memory_gb"],
        }
    )
    predictions = build_predictions_table(
        image_ids=result["image_ids"],
        targets=result["targets"],
        probabilities=result["probabilities"],
        findings=findings,
        threshold=args.threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{args.split}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions.to_csv(
        output_dir / f"{args.split}_predictions.csv",
        index=False,
    )
    evaluation_metadata = {
        "split": args.split,
        "threshold": args.threshold,
        "num_images": len(dataset),
        "model_name": config["model"]["name"],
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "split_assignment_sha256": assignment,
        "amp_enabled": amp_enabled,
        "data_loader": loader_kwargs,
        "findings": findings,
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(evaluation_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
