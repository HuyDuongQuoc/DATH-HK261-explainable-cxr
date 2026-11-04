import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from src.data.dataset import CXRDataset
from src.data.loader import build_loader_kwargs
from src.data.protocol import (
    load_split_metadata,
    validate_checkpoint_protocol,
)
from src.losses.classification_loss import (
    build_classification_loss,
)
from src.metrics.classification import (
    THRESHOLD_STRATEGY,
    compute_classification_metrics,
    optimize_f1_thresholds,
)
from src.models.factory import build_model
from src.training.engine import run_epoch
from src.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from src.utils.checksums import file_sha256
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
    )

    parser.add_argument(
        "--result-dir",
        required=True,
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(config["seed"])

    findings = config["data"]["target_findings"]

    processed_root = Path(
        config["data"]["processed_root"]
    )

    protocol_root = Path(config["data"]["protocol_dir"])
    split_metadata = load_split_metadata(protocol_root)
    if split_metadata.get("protocol_id") != config["data"]["protocol_id"]:
        raise ValueError("Configured data protocol does not match split metadata")
    split_assignment_sha256 = split_metadata["assignment_sha256"]
    config["data"]["split_assignment_sha256"] = split_assignment_sha256

    preprocessing_config = config["preprocessing"]

    train_dataset = CXRDataset(
        manifest_path=config["data"]["train_manifest_path"],
        processed_root=processed_root,
        findings=findings,
        image_size=preprocessing_config["image_size"],
        mean=preprocessing_config["mean"],
        std=preprocessing_config["std"],
        load_masks=False,
    )

    val_dataset = CXRDataset(
        manifest_path=config["data"]["val_manifest_path"],
        processed_root=processed_root,
        findings=findings,
        image_size=preprocessing_config["image_size"],
        mean=preprocessing_config["mean"],
        std=preprocessing_config["std"],
        load_masks=False,
    )
    validation_manifest_sha256 = file_sha256(
        config["data"]["val_manifest_path"]
    )

    training_config = config["training"]
    loader_kwargs = build_loader_kwargs(
        training_config,
        pin_memory=torch.cuda.is_available(),
    )

    train_generator = (
        torch.Generator()
        .manual_seed(config["seed"])
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_model(
        model_name=config["model"]["name"],
        num_classes=len(findings),
        pretrained=config["model"]["pretrained"],
    ).to(device)
    amp_enabled = bool(
        training_config["use_amp"] and device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    config["training"]["amp_active"] = amp_enabled
    config["training"]["device"] = str(device)

    train_labels = (
        train_dataset.table[findings].to_numpy(dtype=float)
    )

    criterion, pos_weight = (
        build_classification_loss(
            train_labels=train_labels,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    result_dir = Path(
        args.result_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_epoch = 0
    best_macro_auroc = float("-inf")
    bad_epochs = 0
    history = []

    if args.resume is not None:
        checkpoint = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        validate_checkpoint_protocol(
            checkpoint=checkpoint,
            model_name=config["model"]["name"],
            findings=findings,
            assignment_sha256=split_assignment_sha256,
            protocol_id=config["data"]["protocol_id"],
        )
        if checkpoint.get("amp_enabled") != amp_enabled:
            raise ValueError("Checkpoint AMP protocol mismatch")
        if checkpoint.get("threshold_strategy") != THRESHOLD_STRATEGY:
            raise ValueError("Checkpoint threshold strategy mismatch")
        if "decision_thresholds" not in checkpoint:
            raise ValueError("Checkpoint is missing decision_thresholds")
        if (
            checkpoint.get("threshold_validation_manifest_sha256")
            != validation_manifest_sha256
        ):
            raise ValueError("Checkpoint validation manifest mismatch")

        start_epoch = checkpoint["epoch"] + 1

        best_macro_auroc = checkpoint["best_macro_auroc"]

        bad_epochs = checkpoint["bad_epochs"]

        history = checkpoint["history"]

    (
        result_dir / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for epoch in range(
        start_epoch,
        training_config["epochs"],
    ):
        if bad_epochs >= training_config[ "patience"]:
            break

        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=amp_enabled,
            show_progress=training_config["show_progress"],
            description=(
                f"Epoch {epoch + 1:02d}/{training_config['epochs']:02d} — train"
            ),
        )

        val_result = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=amp_enabled,
            show_progress=training_config["show_progress"],
            description=(
                f"Epoch {epoch + 1:02d}/{training_config['epochs']:02d} — val"
            ),
        )

        decision_thresholds = optimize_f1_thresholds(
            targets=val_result["targets"],
            probabilities=val_result["probabilities"],
            findings=findings,
        )
        val_metrics = compute_classification_metrics(
            targets=val_result["targets"],
            probabilities=val_result["probabilities"],
            findings=findings,
            threshold=decision_thresholds,
        )
        threshold_values = list(decision_thresholds.values())

        current_score = val_metrics["macro_auroc"]

        improved = current_score > best_macro_auroc + training_config["min_delta"]

        if improved:
            best_macro_auroc = (current_score)
            bad_epochs = 0
        else:
            bad_epochs += 1

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "val_macro_auroc": (current_score),
            "val_macro_ap": val_metrics["macro_average_precision"],
            "val_macro_f1_tuned": val_metrics["macro_f1"],
            "threshold_min": min(threshold_values),
            "threshold_mean": sum(threshold_values) / len(threshold_values),
            "threshold_max": max(threshold_values),
            "train_seconds": train_result["elapsed_seconds"],
            "val_seconds": val_result["elapsed_seconds"],
            "train_images_per_second": train_result["images_per_second"],
            "val_images_per_second": val_result["images_per_second"],
            "peak_gpu_memory_gb": max(
                train_result["peak_gpu_memory_gb"],
                val_result["peak_gpu_memory_gb"],
            ),
        }

        history.append(epoch_record)

        checkpoint = {
            "epoch": epoch,
            "model_name": config["model"]["name"],
            "model_state_dict":  model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_macro_auroc": best_macro_auroc,
            "bad_epochs": bad_epochs,
            "findings": findings,
            "split_assignment_sha256": split_assignment_sha256,
            "data_protocol_id": config["data"]["protocol_id"],
            "amp_enabled": amp_enabled,
            "amp_scaler_state_dict": scaler.state_dict(),
            "pos_weight": pos_weight.detach().cpu(),
            "decision_thresholds": decision_thresholds,
            "threshold_strategy": THRESHOLD_STRATEGY,
            "threshold_validation_manifest_sha256": validation_manifest_sha256,
            "tuned_validation_macro_f1": val_metrics["macro_f1"],
            "config": config,
            "history": history,
        }

        if improved:
            best_metrics_payload = {
                **val_metrics,
                "val_loss": float(val_result["loss"]),
                "checkpoint_epoch": epoch + 1,
                "checkpoint_file": "best.pt",
                "decision_thresholds": decision_thresholds,
                "threshold_strategy": THRESHOLD_STRATEGY,
                "threshold_validation_manifest_sha256": validation_manifest_sha256,
                "amp_enabled": amp_enabled,
                "split_assignment_sha256": split_assignment_sha256,
                "data_protocol_id": config["data"]["protocol_id"],
                "val_seconds": val_result["elapsed_seconds"],
                "val_images_per_second": val_result["images_per_second"],
            }
            (
                result_dir / "best_val_metrics.json"
            ).write_text(
                json.dumps(
                    best_metrics_payload,
                    indent=2,
                ),
                encoding="utf-8",
            )
            save_checkpoint(
                checkpoint,
                checkpoint_dir / "best.pt",
            )

        save_checkpoint(
            checkpoint,
            checkpoint_dir / "last.pt",
        )

        pd.DataFrame(history).to_csv(
            result_dir / "history.csv",
            index=False,
        )

        (
            result_dir
            / "latest_val_metrics.json"
        ).write_text(
            json.dumps(
                {
                    **val_metrics,
                    "decision_thresholds": decision_thresholds,
                    "threshold_strategy": THRESHOLD_STRATEGY,
                    "threshold_validation_manifest_sha256": validation_manifest_sha256,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(epoch_record, flush=True)

    print(
        "Training completed. "
        f"Best macro-AUROC: "
        f"{best_macro_auroc:.4f}"
    )


if __name__ == "__main__":
    main()
