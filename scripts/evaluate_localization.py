import argparse
import json
from pathlib import Path
import sys

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
from src.metrics.localization import iou_column, select_iou_threshold
from src.models.factory import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.xai.evaluation import (
    collect_localization_records,
    summarize_by_finding,
)


def build_loader(config, findings, processed_root, manifest_path):
    preprocessing = config["preprocessing"]
    dataset = CXRDataset(
        manifest_path=manifest_path,
        processed_root=processed_root,
        findings=findings,
        image_size=preprocessing["image_size"],
        mean=preprocessing["mean"],
        std=preprocessing["std"],
        load_masks=True,
    )
    kwargs = build_loader_kwargs(
        config["training"],
        pin_memory=torch.cuda.is_available(),
        batch_size=config["training"]["validation_batch_size"],
    )
    return DataLoader(dataset, shuffle=False, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Grad-CAM localization (Pointing Game, IoU, EIB). IoU threshold "
            "is selected on validation and then fixed for the test split."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Chỉ đánh giá khoảng N ảnh đầu mỗi split (smoke test).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    findings = config["data"]["target_findings"]
    data_config = config["data"]

    split_metadata = load_split_metadata(Path(data_config["protocol_dir"]))
    if split_metadata.get("protocol_id") != data_config["protocol_id"]:
        raise ValueError("Configured data protocol does not match split metadata")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]["name"]
    model = build_model(
        model_name=model_name,
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
        model_name=model_name,
        findings=findings,
        assignment_sha256=split_metadata["assignment_sha256"],
        protocol_id=data_config["protocol_id"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Validation: chọn threshold IoU
    val_loader = build_loader(
        config,
        findings,
        Path(data_config["processed_root"]),
        Path(data_config["val_manifest_path"]),
    )
    val_records = collect_localization_records(
        model, model_name, val_loader, findings, device,
        max_images=args.max_images,
    )
    if val_records.empty:
        raise ValueError("No valid (image, class) pairs in validation")
    threshold, mean_iou_by_threshold = select_iou_threshold(val_records)
    print(f"IoU threshold selected on validation: {threshold:.2f}", flush=True)

    result = {
        "model": model_name,
        "checkpoint": str(args.checkpoint),
        "iou_threshold": threshold,
        "iou_threshold_selected_on": "validation",
        "validation_mean_iou_by_threshold": mean_iou_by_threshold,
        "max_images": args.max_images,
    }

    splits = {"val": (val_records, "validation")}

    # 2) Test: dùng threshold đã cố định
    if args.split == "test":
        external = data_config["external_test"]
        test_loader = build_loader(
            config,
            findings,
            Path(external["processed_root"]),
            Path(external["manifest_path"]),
        )
        test_records = collect_localization_records(
            model, model_name, test_loader, findings, device,
            max_images=args.max_images,
        )
        splits["test"] = (test_records, "external_test")

    for split_name, (records, label) in splits.items():
        table = summarize_by_finding(records, findings, threshold)
        records.to_csv(output_dir / f"{split_name}_localization_records.csv", index=False)
        table.to_csv(output_dir / f"{split_name}_localization_by_finding.csv", index=False)
        macro = table[table["finding"] == "MACRO"].iloc[0]
        result[split_name] = {
            "split": label,
            "n_pairs": int(macro["n_pairs"]),
            "macro_pointing_hit_rate": float(macro["pointing_hit_rate"]),
            "macro_eib": float(macro["eib"]),
            "macro_iou": float(macro["iou"]),
            "iou_column": iou_column(threshold),
        }
        print(split_name, result[split_name], flush=True)

    (output_dir / "localization_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
