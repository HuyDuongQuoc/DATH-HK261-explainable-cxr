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

CHẠY MA TRẬN CONFIG (vd ablation classification loss, mỗi (config, seed) một container):
    ~/.venvs/modal/bin/python -m modal run scripts/modal_sweep.py::ablation \
        --configs loss_r50_raw,loss_r50_sqrtcap3,loss_r50_cb,loss_r50_asl,loss_r50_ignore_minority,loss_r50_soft_votes \
        --seeds 42,43
    ~/.venvs/modal/bin/python -m modal run scripts/modal_sweep.py::test --name loss_r50_asl_base_s42 \
        --lam 0 --seed 42 --config loss_r50_asl
    Tổng hợp sau khi tải về: python scripts/summarize_runs.py outputs/modal_runs

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
            # giữ annotation CSV (cần cho label_policy), bỏ thư mục ảnh gốc
            "data/raw/train/**",
            "data/raw/test/**",
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
# Prior giải phẫu build một lần từ mask train, dùng chung giữa các container.
PRIOR_CACHE = f"{RUNS_ROOT}/priors/train_union_prior.npz"


def _write_config(config_dir: Path, model_name: str, lam: float, seed: int, overlay: str = ""):
    """
    default.yaml (đường dẫn trỏ tới volume) + run.yaml. load_config đọc default.yaml cùng thư mục.

    overlay: tên file trong /app/configs (vd "loss_r50_asl" hoặc "loss_r50_asl.yaml");
    nội dung được dùng làm run.yaml, model_name chỉ dùng khi overlay không đặt model.
    """
    import yaml

    config = yaml.safe_load(open("/app/configs/default.yaml", encoding="utf-8"))
    data = config["data"]
    data["protocol_dir"] = "/app/" + data["protocol_dir"]
    data["processed_root"] = f"{DATA_ROOT}/trainval"
    data["train_manifest_path"] = f"{DATA_ROOT}/train.csv"
    data["val_manifest_path"] = f"{DATA_ROOT}/val.csv"
    data["external_test"]["processed_root"] = f"{DATA_ROOT}/test"
    data["external_test"]["manifest_path"] = f"{DATA_ROOT}/test.csv"
    data["annotation_csv"] = "/app/data/raw/train.csv"
    data["external_test"]["annotation_csv"] = "/app/data/raw/test.csv"
    config["seed"] = int(seed)
    config["model"]["pretrained"] = True
    training = config["training"]
    # Mặc định của protocol Modal; config overlay (vd vit_ft.yaml) có thể ghi đè
    # epochs/patience/learning_rate vì run.yaml được merge sau default.yaml.
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
    run = {}
    if overlay:
        name = overlay if overlay.endswith(".yaml") else f"{overlay}.yaml"
        run = yaml.safe_load(open(f"/app/configs/{name}", encoding="utf-8")) or {}
    run.setdefault("model", {}).setdefault("name", model_name)
    run["model"]["pretrained"] = True
    if lam > 0:
        run.setdefault("localization_loss", {}).update(enabled=True, lambda_loc=lam)
    elif "localization_loss" in run:
        run["localization_loss"]["enabled"] = False
    (config_dir / "run.yaml").write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
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
    # ViT với công thức fine-tune (30 epoch) có thể vượt 2 giờ; run bị ngắt sẽ resume từ last.pt
    timeout=8 * 3600,
    volumes={DATA_ROOT: data_volume, RUNS_ROOT: runs_volume},
)
def train_one(model_name: str, lam: float, seed: int, prefix: str, overlay: str = ""):
    """Train + đánh giá localization trên VALIDATION. Không đụng test."""
    name = run_name(prefix, lam, seed)
    out = Path(RUNS_ROOT) / name
    if (out / "loc" / "localization_summary.json").exists():
        return f"skip {name} (đã có kết quả)"

    config = _write_config(Path(f"/tmp/cfg_{name}"), model_name, lam, seed, overlay)
    ckpt_dir, res_dir, loc_dir = out / "ckpt", out / "res", out / "loc"
    trained = (ckpt_dir / "best.pt").is_file() and not (ckpt_dir / "last.pt").exists()
    if trained:
        print(f"{name}: đã train xong (best.pt, không còn last.pt), chỉ chạy đánh giá", flush=True)
    else:
        cmd = [
            sys.executable, "scripts/train.py", "--config", str(config),
            "--checkpoint-dir", str(ckpt_dir), "--result-dir", str(res_dir),
        ]
        if (ckpt_dir / "last.pt").is_file():
            cmd += ["--resume", str(ckpt_dir / "last.pt")]  # container trước bị ngắt giữa chừng
        _run(cmd)
        (ckpt_dir / "last.pt").unlink(missing_ok=True)  # đánh dấu train xong, chỉ giữ best.pt
        runs_volume.commit()
    _run([
        sys.executable, "scripts/evaluate_localization.py", "--config", str(config),
        "--checkpoint", str(ckpt_dir / "best.pt"), "--output-dir", str(loc_dir),
        "--split", "val", "--prior-cache", PRIOR_CACHE,
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
def test_one(name: str, model_name: str, lam: float, seed: int, overlay: str = ""):
    """Đánh giá EXTERNAL TEST (classification + localization) cho một run đã có best.pt."""
    runs_volume.reload()
    out = Path(RUNS_ROOT) / name
    checkpoint = out / "ckpt" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = _write_config(Path(f"/tmp/cfg_{name}"), model_name, lam, seed, overlay)
    _run([
        sys.executable, "scripts/evaluate.py", "--config", str(config),
        "--checkpoint", str(checkpoint), "--output-dir", str(out / "cls_test"),
        "--split", "test",
    ])
    _run([
        sys.executable, "scripts/evaluate_localization.py", "--config", str(config),
        "--checkpoint", str(checkpoint), "--output-dir", str(out / "loc_test"),
        "--split", "test", "--prior-cache", PRIOR_CACHE,
    ])
    runs_volume.commit()
    return f"tested {name}"


def _config_from_saved_run(out: Path) -> Path:
    """
    Dựng lại config đúng như lúc train từ <run>/res/config.json (train.py lưu cấu hình
    đã merge). Ghi thành default.yaml để load_config đọc trực tiếp, không cần biết
    model/lambda/loss của run.
    """
    import json

    import yaml

    saved = json.loads((out / "res" / "config.json").read_text(encoding="utf-8"))
    data = saved["data"]
    # run train local (vd RTX 3060) lưu đường dẫn máy local: trỏ lại về volume
    data["processed_root"] = f"{DATA_ROOT}/trainval"
    data["train_manifest_path"] = f"{DATA_ROOT}/train.csv"
    data["val_manifest_path"] = f"{DATA_ROOT}/val.csv"
    data.setdefault("external_test", {})
    data["external_test"]["processed_root"] = f"{DATA_ROOT}/test"
    data["external_test"]["manifest_path"] = f"{DATA_ROOT}/test.csv"
    # annotation CSV nằm trong image tại /app/data/raw
    data["annotation_csv"] = "/app/data/raw/train.csv"
    data.setdefault("external_test", {})["annotation_csv"] = "/app/data/raw/test.csv"
    data["protocol_dir"] = f"/app/data/protocols/{data['protocol_id']}"
    data["validation_ids_path"] = f"{data['protocol_dir']}/validation_ids.csv"
    config_dir = Path(f"/tmp/cfg_eval_{out.name}")
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "default.yaml"
    path.write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")
    return path


# Ngưỡng IoU cố định chung cho mọi run khi so sánh (tránh chọn ngưỡng riêng từng run).
COMMON_IOU_THRESHOLD = "0.30"


@app.function(
    image=image,
    cpu=8,
    memory=24576,
    timeout=2 * 3600,
    volumes={DATA_ROOT: data_volume, RUNS_ROOT: runs_volume},
)
def build_prior(name: str):
    """Build prior giải phẫu một lần (CPU) trước khi nhiều container GPU cùng dùng."""
    sys.path.insert(0, "/app")
    from src.utils.config import load_config
    from src.xai.priors import accumulate_class_priors, load_or_build_priors
    from scripts.evaluate_localization import build_loader

    runs_volume.reload()
    if Path(PRIOR_CACHE).is_file():
        return f"prior đã có: {PRIOR_CACHE}"
    config = load_config(_config_from_saved_run(Path(RUNS_ROOT) / name))
    findings = config["data"]["target_findings"]
    loader = build_loader(
        config, findings,
        Path(config["data"]["processed_root"]), Path(config["data"]["train_manifest_path"]),
    )
    load_or_build_priors(PRIOR_CACHE, lambda: accumulate_class_priors(loader, len(findings)), findings)
    runs_volume.commit()
    return f"built {PRIOR_CACHE}"


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=24576,
    timeout=3 * 3600,
    volumes={DATA_ROOT: data_volume, RUNS_ROOT: runs_volume},
)
def eval_one(name: str, occlusion: bool = True):
    """
    Đánh giá EXTERNAL TEST cho một run đã train, dùng đúng config đã lưu:
      cls_test/               classification (bỏ qua nếu đã có)
      loc_test_gradcam/       Grad-CAM + prior + box_area, ngưỡng IoU cố định
      loc_test_occlusion/     occlusion + box-occlusion (độc lập với L_loc), nếu occlusion=True
    Mỗi bước bỏ qua nếu đã có localization_summary.json / test_metrics.json.
    """
    runs_volume.reload()
    out = Path(RUNS_ROOT) / name
    checkpoint = out / "ckpt" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = _config_from_saved_run(out)
    done = []

    if not (out / "cls_test" / "test_metrics.json").is_file():
        _run([
            sys.executable, "scripts/evaluate.py", "--config", str(config),
            "--checkpoint", str(checkpoint), "--output-dir", str(out / "cls_test"),
            "--split", "test",
        ])
        runs_volume.commit()
        done.append("cls")

    methods = [("loc_test_gradcam", ["--method", "gradcam"])]
    if occlusion:
        methods.append(("loc_test_occlusion", ["--method", "occlusion", "--box-occlusion"]))
    for folder, extra in methods:
        if (out / folder / "localization_summary.json").is_file():
            continue
        _run([
            sys.executable, "scripts/evaluate_localization.py", "--config", str(config),
            "--checkpoint", str(checkpoint), "--output-dir", str(out / folder),
            "--split", "test", "--prior-cache", PRIOR_CACHE,
            "--iou-threshold", COMMON_IOU_THRESHOLD, *extra,
        ])
        runs_volume.commit()
        done.append(folder)
    return f"evaluated {name}: {done or 'đã đủ, không chạy lại'}"


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
    _report(train_one.starmap(jobs, return_exceptions=True))


@app.local_entrypoint()
def ablation(
    configs: str,
    seeds: str = "42,43",
    lam: float = 0.0,
    model: str = "resnet50",
    max_jobs: int = 12,
):
    """
    Mỗi (config, seed) một container. configs: tên file trong configs/ (không cần .yaml),
    phân tách bằng dấu phẩy. Tên run = <config>_<base|lX>_s<seed>.
    """
    config_names = [c.strip().removesuffix(".yaml") for c in configs.split(",") if c.strip()]
    missing = [c for c in config_names if not (REPO / "configs" / f"{c}.yaml").is_file()]
    if missing:
        raise SystemExit(f"Không tìm thấy config: {missing}")
    seed_values = [int(x) for x in seeds.split(",")]
    jobs = [(model, lam, seed, c, c) for c in config_names for seed in seed_values]
    if len(jobs) > max_jobs:
        raise SystemExit(
            f"{len(jobs)} job vượt --max-jobs {max_jobs}. Tăng --max-jobs nếu bạn chủ ý, "
            "sau khi đã kiểm tra chi phí."
        )
    print(f"Chạy {len(jobs)} job song song trên GPU {GPU}: {[run_name(j[3], lam, j[2]) for j in jobs]}")
    _report(train_one.starmap(jobs, return_exceptions=True))


def _report(results):
    """In kết quả từng job; một job lỗi không làm dừng các job còn lại."""
    failed = 0
    for message in results:
        if isinstance(message, BaseException):
            failed += 1
            print(f"FAILED: {type(message).__name__}: {message}", flush=True)
        else:
            print(message, flush=True)
    print(f"Xong: {failed} job lỗi", flush=True)


@app.local_entrypoint()
def test(
    name: str,
    lam: float,
    seed: int = 42,
    model: str = "resnet50",
    config: str = "",
):
    """
    Test cho MỘT run đã chọn. Truyền lambda rõ ràng (không suy ngược từ tên run);
    --config phải trùng config overlay đã dùng khi train (nếu có).
    """
    print(test_one.remote(name, model, lam, seed, config))


@app.local_entrypoint()
def evaluate_runs(runs: str, occlusion: bool = True, max_jobs: int = 12):
    """
    Đánh giá external test (classification, Grad-CAM, occlusion) cho các run đã train,
    mỗi run một container. Config đọc lại từ <run>/res/config.json.
        modal run scripts/modal_sweep.py::evaluate_runs --runs r50_base_s42,d121_base_s42
    """
    names = [r.strip() for r in runs.split(",") if r.strip()]
    if not names:
        raise SystemExit("--runs rỗng")
    if len(names) > max_jobs:
        raise SystemExit(f"{len(names)} job vượt --max-jobs {max_jobs}")
    print(build_prior.remote(names[0]), flush=True)
    print(f"Đánh giá {len(names)} run trên GPU {GPU}: {names}", flush=True)
    _report(eval_one.starmap([(n, occlusion) for n in names], return_exceptions=True))
