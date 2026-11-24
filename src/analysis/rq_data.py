"""
Đọc kết quả của các run (đã tải từ volume Modal hoặc chạy local) thành bảng dài
run × finding để phân tích RQ1/RQ2. Không import matplotlib.

Bố cục một run (tạo bởi scripts/modal_sweep.py):
    <run>/res/config.json, res/best_val_metrics.json
    <run>/cls_test/test_metrics.json
    <run>/loc/                 Grad-CAM trên validation (train_one)
    <run>/loc_test_gradcam/    Grad-CAM trên test, có prior/box_area (eval_one)
    <run>/loc_test_occlusion/  occlusion + box-occlusion trên test (eval_one)
    <run>/loc_test/            bản cũ (Grad-CAM, chưa có prior)
"""

import json
import re
from pathlib import Path

import pandas as pd

ARCH_LABELS = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "vit_b_16": "ViT-B/16",
    "vit_b_16_gap": "ViT-B/16 (GAP)",
}

LOC_FOLDERS = {
    ("test", "gradcam"): ["loc_test_gradcam", "loc_test"],
    ("test", "occlusion"): ["loc_test_occlusion"],
    ("val", "gradcam"): ["loc"],
}

SEED_PATTERN = re.compile(r"_s(\d+)$")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_info(run_dir):
    """model, lambda (0 nếu tắt L_loc), seed, loss của một run, đọc từ config đã lưu."""
    run_dir = Path(run_dir)
    config = _read_json(run_dir / "res" / "config.json") or _read_json(run_dir / "config.json")
    if config is None:
        return None
    loc = config.get("localization_loss", {})
    lam = float(loc.get("lambda_loc", 0.0)) if loc.get("enabled") else 0.0
    seed = config.get("seed")
    if seed is None:
        match = SEED_PATTERN.search(run_dir.name)
        seed = int(match.group(1)) if match else None
    return {
        "run": run_dir.name,
        "model": config["model"]["name"],
        "arch": ARCH_LABELS.get(config["model"]["name"], config["model"]["name"]),
        "lambda": lam,
        "seed": seed,
        "loss": json.dumps(config.get("training", {}).get("loss", {}), sort_keys=True),
    }


def classification_by_finding(run_dir, split="test"):
    run_dir = Path(run_dir)
    if split == "test":
        metrics = _read_json(run_dir / "cls_test" / "test_metrics.json")
    else:
        metrics = _read_json(run_dir / "res" / "best_val_metrics.json") or _read_json(
            run_dir / "best_val_metrics.json"
        )
    if not metrics:
        return None
    rows = metrics["per_class"]
    frame = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame(
        [{"finding": k, **v} for k, v in rows.items()]
    )
    return frame[["finding", "auroc", "average_precision"]].rename(
        columns={"average_precision": "auprc"}
    )


def localization_folder(run_dir, split="test", method="gradcam"):
    for folder in LOC_FOLDERS[(split, method)]:
        path = Path(run_dir) / folder
        if (path / f"{split}_localization_by_finding.csv").is_file():
            return path
    return None


def localization_by_finding(run_dir, split="test", method="gradcam"):
    folder = localization_folder(run_dir, split, method)
    if folder is None:
        return None
    table = pd.read_csv(folder / f"{split}_localization_by_finding.csv")
    return table[table["finding"] != "MACRO"].copy()


def localization_records(run_dir, split="test", method="gradcam"):
    folder = localization_folder(run_dir, split, method)
    if folder is None:
        return None
    path = folder / f"{split}_localization_records.csv"
    return pd.read_csv(path) if path.is_file() else None


def predictions_path(run_dir, split="test"):
    run_dir = Path(run_dir)
    candidates = (
        [run_dir / "cls_test" / "test_predictions.csv"]
        if split == "test"
        else [run_dir / "res" / "best_val_predictions.csv", run_dir / "best_val_predictions.csv"]
    )
    return next((p for p in candidates if p.is_file()), None)


def collect_runs(root, pattern="*"):
    infos = []
    for run_dir in sorted(Path(root).glob(pattern)):
        if run_dir.is_dir():
            info = run_info(run_dir)
            if info is not None:
                info["path"] = str(run_dir)
                infos.append(info)
    return pd.DataFrame(infos)


def long_table(runs, split="test", method="gradcam"):
    """Bảng run × finding gồm metric classification và localization (bỏ run thiếu dữ liệu)."""
    frames, missing = [], []
    for info in runs.to_dict("records"):
        cls = classification_by_finding(info["path"], split)
        loc = localization_by_finding(info["path"], split, method)
        if cls is None or loc is None:
            missing.append(info["run"])
            continue
        merged = cls.merge(loc, on="finding", how="inner")
        for key in ("run", "model", "arch", "lambda", "seed"):
            merged[key] = info[key]
        frames.append(merged)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return table, missing


def markdown_table(frame, index=True, digits=4):
    """Bảng Markdown đơn giản (không cần thư viện tabulate)."""
    frame = frame.round(digits)
    if index:
        frame = frame.reset_index()
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])
