"""
Đánh giá external test cho các run đã train (thay cho modal_sweep.py::evaluate_runs
khi không dùng được Modal), chạy được trên Kaggle hoặc bất kỳ máy có GPU nào.

Bố cục bundle (tạo bởi scripts/make_eval_bundle.py):
    <bundle>/bundle_manifest.json
    <bundle>/code/{src,scripts,configs,data/protocols,data/raw/*.csv}
    <bundle>/data/{trainval/,test/,train.csv,val.csv,test.csv}
    <bundle>/priors/train_union_prior.npz
    <bundle>/runs/<run>/{ckpt/best.pt,res/config.json}

    python <bundle>/code/scripts/kaggle_eval.py --bundle <bundle> --output /kaggle/working

Kết quả cho mỗi run (bỏ qua bước đã có, nên chạy lại được):
    <output>/runs/<run>/res/config.json
    <output>/runs/<run>/cls_test/              classification
    <output>/runs/<run>/loc_test_gradcam/      Grad-CAM + prior, IoU 0.30, chỉ test
    <output>/runs/<run>/loc_test_occlusion/    occlusion + box-occlusion, chỉ test
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

COMMON_IOU_THRESHOLD = "0.30"


def remap_saved_config(saved, data_root, code_root, num_workers=2):
    """Trỏ đường dẫn trong config đã lưu (Modal/local) về bundle; số worker theo máy đánh giá."""
    training = saved.setdefault("training", {})
    training["num_workers"] = int(num_workers)
    training["persistent_workers"] = False
    training["show_progress"] = False
    data = saved["data"]
    data_root, code_root = Path(data_root), Path(code_root)
    data["processed_root"] = str(data_root / "trainval")
    data["train_manifest_path"] = str(data_root / "train.csv")
    data["val_manifest_path"] = str(data_root / "val.csv")
    external = data.setdefault("external_test", {})
    external["processed_root"] = str(data_root / "test")
    external["manifest_path"] = str(data_root / "test.csv")
    external["annotation_csv"] = str(code_root / "data" / "raw" / "test.csv")
    data["annotation_csv"] = str(code_root / "data" / "raw" / "train.csv")
    data["protocol_dir"] = str(code_root / "data" / "protocols" / data["protocol_id"])
    data["validation_ids_path"] = str(Path(data["protocol_dir"]) / "validation_ids.csv")
    return saved


def run(cmd, cwd):
    print("$", " ".join(cmd), flush=True)
    started = time.time()
    env = {**os.environ, "PYTHONPATH": str(cwd), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(cmd, check=True, cwd=cwd, env=env)
    print(f"  ({time.time() - started:.0f}s)", flush=True)


def evaluate_run(name, bundle, output, occlusion, tmp_root, max_images=None, num_workers=2):
    code = bundle / "code"
    run_dir = bundle / "runs" / name
    checkpoint = run_dir / "ckpt" / "best.pt"
    saved = json.loads((run_dir / "res" / "config.json").read_text(encoding="utf-8"))
    out = output / "runs" / name
    (out / "res").mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "res" / "config.json", out / "res" / "config.json")

    config_dir = tmp_root / f"cfg_{name}"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = config_dir / "default.yaml"
    config.write_text(
        yaml.safe_dump(remap_saved_config(saved, bundle / "data", code, num_workers), sort_keys=False),
        encoding="utf-8",
    )

    done = []
    if max_images is None and not (out / "cls_test" / "test_metrics.json").is_file():
        run([sys.executable, "scripts/evaluate.py", "--config", str(config),
             "--checkpoint", str(checkpoint), "--output-dir", str(out / "cls_test"),
             "--split", "test"], cwd=code)
        done.append("cls")

    methods = [("loc_test_gradcam", ["--method", "gradcam"])]
    if occlusion:
        methods.append(("loc_test_occlusion", ["--method", "occlusion", "--box-occlusion"]))
    for folder, extra in methods:
        if (out / folder / "localization_summary.json").is_file():
            continue
        run([sys.executable, "scripts/evaluate_localization.py", "--config", str(config),
             "--checkpoint", str(checkpoint), "--output-dir", str(out / folder),
             "--split", "test", "--test-only", "--iou-threshold", COMMON_IOU_THRESHOLD,
             "--prior-cache", str(bundle / "priors" / "train_union_prior.npz"), *extra,
             *(["--max-images", str(max_images), "--n-boot", "50"] if max_images else [])], cwd=code)
        done.append(folder)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", default="", help="Danh sách run (mặc định: mọi run trong bundle)")
    parser.add_argument("--no-occlusion", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--smoke", type=int, default=None,
        help="Chỉ localization trên N ảnh đầu, bỏ classification (kiểm tra đường dẫn/protocol trên CPU)",
    )
    args = parser.parse_args()

    bundle, output = Path(args.bundle).resolve(), Path(args.output).resolve()
    names = [r.strip() for r in args.runs.split(",") if r.strip()] or sorted(
        p.name for p in (bundle / "runs").iterdir() if (p / "ckpt" / "best.pt").is_file()
    )
    tmp_root = output / "tmp_configs"
    failures = []
    for name in names:
        try:
            print(f"== {name}: {evaluate_run(
                name, bundle, output, not args.no_occlusion, tmp_root, args.smoke, args.num_workers
            )}", flush=True)
        except subprocess.CalledProcessError as error:
            failures.append(name)
            print(f"FAILED {name}: {error}", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"Xong: {len(failures)} run lỗi {failures}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
