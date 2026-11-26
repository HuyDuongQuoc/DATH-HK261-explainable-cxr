"""
Đóng gói code + dữ liệu đã tiền xử lý (validation + external test) + checkpoint +
prior thành một thư mục để upload làm Kaggle dataset (private) rồi đánh giá bằng
scripts/kaggle_eval.py.

    python scripts/make_eval_bundle.py --source outputs/kaggle_bundle --output outputs/kaggle_dataset

--source phải có data/{trainval,test,train.csv,val.csv,test.csv}, priors/, runs/<run>/...
(tải từ volume Modal). Chỉ giữ ảnh/mask của validation trong trainval, và bỏ trạng
thái optimizer khỏi checkpoint để giảm dung lượng upload.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ITEMS = ["src", "scripts", "configs", "data/protocols"]
RAW_CSV = ["data/raw/train.csv", "data/raw/test.csv"]
DROP_CHECKPOINT_KEYS = ("optimizer_state_dict", "amp_scaler_state_dict")


def copy_code(dest):
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for item in CODE_ITEMS:
        shutil.copytree(PROJECT_ROOT / item, dest / item, ignore=ignore, dirs_exist_ok=True)
    for item in RAW_CSV:
        (dest / item).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PROJECT_ROOT / item, dest / item)


def copy_split(source_root, manifest, dest_root):
    table = pd.read_csv(manifest)
    for column in ("image_path", "mask_path"):
        for rel in table[column]:
            target = dest_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_root / rel, target)
    return len(table)


def slim_checkpoint(source, dest):
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    removed = [k for k in DROP_CHECKPOINT_KEYS if checkpoint.pop(k, None) is not None]
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dest)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        shutil.rmtree(output)

    copy_code(output / "code")
    data = output / "data"
    data.mkdir(parents=True)
    for name in ("train.csv", "val.csv", "test.csv"):
        shutil.copy(source / "data" / name, data / name)
    n_val = copy_split(source / "data" / "trainval", source / "data" / "val.csv", data / "trainval")
    n_test = copy_split(source / "data" / "test", source / "data" / "test.csv", data / "test")
    shutil.copytree(source / "priors", output / "priors")

    runs = []
    for run_dir in sorted((source / "runs").iterdir()):
        checkpoint = run_dir / "ckpt" / "best.pt"
        if not checkpoint.is_file():
            continue
        removed = slim_checkpoint(checkpoint, output / "runs" / run_dir.name / "ckpt" / "best.pt")
        (output / "runs" / run_dir.name / "res").mkdir(parents=True, exist_ok=True)
        shutil.copy(run_dir / "res" / "config.json", output / "runs" / run_dir.name / "res" / "config.json")
        runs.append({"run": run_dir.name, "removed_keys": removed})

    manifest = {"n_val_images": n_val, "n_test_images": n_test, "runs": runs}
    (output / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
