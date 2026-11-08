"""
Chạy song song các thí nghiệm location-aware (ResNet50 / bất kỳ model nào trong config) trên Modal.

CHUẨN BỊ (một lần, chạy từ thư mục gốc của repo, dùng venv có `modal`):
    ~/.venvs/modal/bin/python -m modal volume create cxr-full-data
    ~/.venvs/modal/bin/python -m modal volume put cxr-full-data outputs/full/data /data
    ~/.venvs/modal/bin/python -m modal volume ls cxr-full-data /data      # kiểm tra: trainval/ test/ train.csv val.csv test.csv

CHẠY (mỗi (lambda, seed) là một container GPU riêng, chạy song song):
    ~/.venvs/modal/bin/python -m modal run scripts/modal_sweep.py::sweep --lambdas 0,0.5,1,2,5 --seeds 42
    ~/.venvs/modal/bin/python -m modal run scripts/modal_sweep.py::test --name r50_l5_s42 --lam 5 --seed 42
    ~/.venvs/modal/bin/python -m modal run scripts/modal_sweep.py::test --name r50_base_s42 --lam 0 --seed 42

LẤY KẾT QUẢ:
    ~/.venvs/modal/bin/python -m modal volume get cxr-runs / outputs/modal_runs

Quy ước: lambda = 0 nghĩa là baseline (không có L_loc). Test chỉ chạy qua entrypoint `test`
cho các run được chọn sau khi đã chọn lambda trên validation.
"""
import os
import subprocess
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
GPU = os.environ.get("MODAL_GPU", "L4")

app = modal.App("cxr-location-aware")
data_volume = modal.Volume.from_name("cxr-full-data", create_if_missing=False)
runs_volume = modal.Volume.from_name("cxr-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "PyYAML",
        "tqdm",
        "Pillow",
        "opencv-python-headless",
    )
    .add_local_dir(
        REPO,
        remote_path="/app",
        ignore=[
            "venv/**",
            "outputs/**",
            ".git/**",
            "data/raw/**",
            "data/processed/**",
            "data/interim/**",
            "tex/**",
            "docs/tex/**",
            "notebooks/**",
            "**/__pycache__",
            "**/*.pt",
        ],
    )
)

DATA_ROOT = "/data"
RUNS_ROOT = "/runs"


def _write_config(config_dir: Path, model_name: str, lam: float, seed: int):
    """default.yaml (đường dẫn trỏ tới volume) + file model. load_config đọc default.yaml cùng thư mục."""
    import yaml

    config = yaml.safe_load(open("/app/configs/default.yaml", encoding="utf-8"))
    data = config["data"]
    data["protocol_dir"] = "/app/" + data["protocol_dir"]
    data["processed_root"] = f"{DATA_ROOT}/trainval"
    data["train_manifest_path"] = f"{DATA_ROOT}/train.csv"
    data["val_manifest_path"] = f"{DATA_ROOT}/val.csv"
    data["external_test"]["processed_root"] = f"{DATA_ROOT}/test"
    data["external_test"]["manifest_path"] = f"{DATA_ROOT}/test.csv"
    config["seed"] = int(seed)
    config["model"]["pretrained"] = True
    training = config["training"]
    training.update(
        batch_size=16,
        validation_batch_size=32,
        num_workers=6,
        epochs=12,
        patience=3,
        show_progress=False,
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    model_yaml = f"model:\n  name: {model_name}\n  pretrained: true\n"
    if lam > 0:
        model_yaml += f"localization_loss:\n  enabled: true\n  lambda_loc: {lam}\n"
    (config_dir / "run.yaml").write_text(model_yaml, encoding="utf-8")
    return config_dir / "run.yaml"


def _run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/app", env={**os.environ, "PYTHONPATH": "/app"})


def run_name(prefix: str, lam: float, seed: int) -> str:
    tag = "base" if lam == 0 else "l" + f"{lam:g}".replace(".", "")
    return f"{prefix}_{tag}_s{seed}"


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=24576,
    timeout=2 * 3600,
    volumes={DATA_ROOT: data_volume, RUNS_ROOT: runs_volume},
)
def train_one(model_name: str, lam: float, seed: int, prefix: str):
    """Train + đánh giá localization trên VALIDATION. Không đụng test."""
    name = run_name(prefix, lam, seed)
    out = Path(RUNS_ROOT) / name
    if (out / "loc" / "localization_summary.json").exists():
        return f"skip {name} (đã có kết quả)"

    config = _write_config(Path(f"/tmp/cfg_{name}"), model_name, lam, seed)
    ckpt_dir, res_dir, loc_dir = out / "ckpt", out / "res", out / "loc"
    _run([
        sys.executable, "scripts/train.py", "--config", str(config),
        "--checkpoint-dir", str(ckpt_dir), "--result-dir", str(res_dir),
    ])
    (ckpt_dir / "last.pt").unlink(missing_ok=True)  # chỉ giữ best.pt để tiết kiệm dung lượng
    _run([
        sys.executable, "scripts/evaluate_localization.py", "--config", str(config),
        "--checkpoint", str(ckpt_dir / "best.pt"), "--output-dir", str(loc_dir),
        "--split", "val",
    ])
    runs_volume.commit()
    return f"done {name}"


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=24576,
    timeout=2 * 3600,
    volumes={DATA_ROOT: data_volume, RUNS_ROOT: runs_volume},
)
def test_one(name: str, model_name: str, lam: float, seed: int):
    """Đánh giá EXTERNAL TEST (classification + localization) cho một run đã có best.pt."""
    runs_volume.reload()
    out = Path(RUNS_ROOT) / name
    checkpoint = out / "ckpt" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = _write_config(Path(f"/tmp/cfg_{name}"), model_name, lam, seed)
    _run([
        sys.executable, "scripts/evaluate.py", "--config", str(config),
        "--checkpoint", str(checkpoint), "--output-dir", str(out / "cls_test"),
        "--split", "test",
    ])
    _run([
        sys.executable, "scripts/evaluate_localization.py", "--config", str(config),
        "--checkpoint", str(checkpoint), "--output-dir", str(out / "loc_test"),
        "--split", "test",
    ])
    runs_volume.commit()
    return f"tested {name}"


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    timeout=3600,
    volumes={DATA_ROOT: data_volume},
)
def unpack_data():
    """Giải nén /data.tar (upload một file nhanh hơn nhiều so với 36.000 file nhỏ)."""
    import tarfile

    data_volume.reload()
    tar_path = Path(DATA_ROOT) / "data.tar"
    with tarfile.open(tar_path) as tar:
        tar.extractall(DATA_ROOT)
    tar_path.unlink()
    data_volume.commit()
    names = sorted(p.name for p in Path(DATA_ROOT).iterdir())
    n_img = len(list((Path(DATA_ROOT) / "trainval" / "images").iterdir()))
    n_test = len(list((Path(DATA_ROOT) / "test" / "images").iterdir()))
    return f"unpacked: {names}; trainval images={n_img}; test images={n_test}"


@app.local_entrypoint()
def unpack():
    print(unpack_data.remote())


@app.local_entrypoint()
def sweep(
    lambdas: str = "0,0.5,1,2,5",
    seeds: str = "42",
    model: str = "resnet50",
    prefix: str = "r50",
    max_jobs: int = 12,
):
    lam_values = [float(x) for x in lambdas.split(",")]
    seed_values = [int(x) for x in seeds.split(",")]
    jobs = [(model, lam, seed, prefix) for seed in seed_values for lam in lam_values]
    if len(jobs) > max_jobs:
        raise SystemExit(
            f"{len(jobs)} job vượt --max-jobs {max_jobs}. Tăng --max-jobs nếu bạn chủ ý, "
            "sau khi đã kiểm tra chi phí."
        )
    print(f"Chạy {len(jobs)} job song song trên GPU {GPU}: {[run_name(prefix, j[1], j[2]) for j in jobs]}")
    for message in train_one.starmap(jobs):
        print(message)


@app.local_entrypoint()
def test(
    name: str,
    lam: float,
    seed: int = 42,
    model: str = "resnet50",
):
    """Test cho MỘT run đã chọn. Truyền lambda rõ ràng (không suy ngược từ tên run)."""
    print(test_one.remote(name, model, lam, seed))
