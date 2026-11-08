import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile

import pandas as pd
import torch
import yaml

from scripts.train import build_augmentation_protocol, build_training_protocol
from src.analysis.stage2 import analyze_stage2
from src.data.protocol_splitting import assignment_sha256
from src.utils.checksums import file_sha256
from src.utils.config import load_config


STAGE1_RUNTIME_CONTRACT = {
    "gpu_name": "Tesla T4",
    "total_memory_mib": 15360,
    "driver": "580.82.07",
    "cpu_count": 2,
    "torch": "2.14.1+cu130",
    "cuda": "13.0",
    "cudnn": 92800,
    "batch_size": 8,
    "validation_batch_size": 64,
    "num_workers": 2,
    "prefetch_factor": 2,
    "amp": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
}
EXPECTED_IMAGES = 15_000
EXPECTED_MASKS = 15_000
EXPECTED_TRAIN_ROWS = 12_750
EXPECTED_VAL_ROWS = 2_250
DATA_PROTOCOL_ID = "vindr_s8515_exttest_v1"
EXPERIMENT_GROUP = "aug_auprc_v1"
NOTEBOOK_GROUP = "03_testing"
STAGE_ID = "stage2_focalloss"


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_or_validate_contract(path, expected):
    """Atomically create the shared contract, or reject any existing mismatch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(expected, indent=2, ensure_ascii=False)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        created = True
    except FileExistsError:
        created = False
    actual = json.loads(path.read_text(encoding="utf-8"))
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"Stage 2 shared contract mismatch: {path}")
    return created


def build_stage2_experiments(model_name, model_slug, config_path, weighting, positive_weight):
    variants = [
        ("bce", "weighted_bce", None, "bce"),
        ("focal_g05", "sigmoid_focal", 0.5, "focal_g05"),
        ("focal_g10", "sigmoid_focal", 1.0, "focal_g10"),
        ("focal_g15", "sigmoid_focal", 1.5, "focal_g15"),
    ]
    experiments = []
    for variant, loss_name, gamma, run_loss in variants:
        loss = {
            "name": loss_name,
            "positive_weight": copy.deepcopy(positive_weight),
        }
        if gamma is not None:
            loss["gamma"] = gamma
        experiments.append({
            "model_name": model_name,
            "model_slug": model_slug,
            "config": config_path,
            "variant": variant,
            "loss_name": loss_name,
            "gamma": gamma,
            "weighting": weighting,
            "positive_weight": copy.deepcopy(positive_weight),
            "loss": loss,
            "run_name": (
                f"{model_slug}_stage2_{run_loss}_{weighting}_"
                "aug_auprc_s8515_exttest_v1_seed42_amp"
            ),
        })
    if len({item["run_name"] for item in experiments}) != 4:
        raise ValueError("Stage 2 run names must be unique")
    return experiments


def _runtime_profile():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    result = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("Stage 2 requires exactly one T4 GPU")
    gpu_name, total_memory_mib, driver = [part.strip() for part in lines[0].split(",")]
    return {
        "gpu_name": gpu_name,
        "total_memory_mib": int(float(total_memory_mib)),
        "driver": driver,
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "batch_size": 8,
        "validation_batch_size": 64,
        "num_workers": 2,
        "prefetch_factor": 2,
        "amp": True,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def _validate_runtime(profile):
    mismatches = {
        key: {"expected": expected, "actual": profile.get(key)}
        for key, expected in STAGE1_RUNTIME_CONTRACT.items()
        if profile.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Runtime does not match the locked Stage 1 contract:\n"
            + json.dumps(mismatches, indent=2)
        )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        probe = torch.ones(8, 8, device="cuda") @ torch.ones(8, 8, device="cuda")
    if not torch.isfinite(probe).all():
        raise RuntimeError("T4 FP16 AMP probe failed")


def _safe_extract(archive_path, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination)
            except ValueError as error:
                raise ValueError(f"Unsafe ZIP member: {member.filename}") from error
        archive.extractall(destination)


def _validate_train_data(root, protocol_root):
    root = Path(root)
    protocol_root = Path(protocol_root)
    required = [root / "images", root / "masks", root / "manifest.csv", root / "metadata.json"]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"Incomplete training archive at {root}")
    train = pd.read_csv(protocol_root / "train_manifest.csv", dtype={"image_id": str})
    val = pd.read_csv(protocol_root / "val_manifest.csv", dtype={"image_id": str})
    manifest = pd.read_csv(root / "manifest.csv", dtype={"image_id": str})
    if len(train) != EXPECTED_TRAIN_ROWS or len(val) != EXPECTED_VAL_ROWS:
        raise ValueError("Split row count mismatch")
    if set(train.image_id) & set(val.image_id):
        raise ValueError("Train/validation overlap")
    if set(train.image_id) | set(val.image_id) != set(manifest.image_id):
        raise ValueError("Split coverage mismatch")
    metadata = json.loads((protocol_root / "split_metadata.json").read_text(encoding="utf-8"))
    if metadata["protocol_id"] != DATA_PROTOCOL_ID:
        raise ValueError("Data protocol ID mismatch")
    if assignment_sha256({"train": train, "val": val}) != metadata["assignment_sha256"]:
        raise ValueError("Split assignment fingerprint mismatch")
    if len(list((root / "images").glob("*.png"))) != EXPECTED_IMAGES:
        raise ValueError("Processed PNG count mismatch")
    if len(list((root / "masks").glob("*.npz"))) != EXPECTED_MASKS:
        raise ValueError("Processed NPZ count mismatch")
    return {
        "data_protocol_id": DATA_PROTOCOL_ID,
        "split_assignment_sha256": metadata["assignment_sha256"],
        "validation_manifest_sha256": file_sha256(protocol_root / "val_manifest.csv"),
    }


def _pip_freeze_sha256():
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    normalized = "\n".join(sorted(line.strip() for line in result.stdout.splitlines() if line.strip())) + "\n"
    return _sha256_text(normalized)


def _build_runtime_config(experiment, repo_dir, processed_root, protocol_root, runtime_root):
    config = load_config(repo_dir / experiment["config"])
    config["data"].update({
        "processed_root": str(processed_root),
        "protocol_dir": str(protocol_root),
        "train_manifest_path": str(protocol_root / "train_manifest.csv"),
        "val_manifest_path": str(protocol_root / "val_manifest.csv"),
    })
    config["training"].update({
        "batch_size": 8,
        "validation_batch_size": 64,
        "num_workers": 2,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "show_progress": False,
        "monitor": "macro_average_precision",
        "loss": copy.deepcopy(experiment["loss"]),
    })
    if config["seed"] != 42:
        raise ValueError("Stage 2 requires seed 42")
    if not config["preprocessing"]["use_augmentation"]:
        raise ValueError("Stage 2 requires the locked augmentation protocol")
    runtime_dir = runtime_root / experiment["run_name"]
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / "default.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    validated = load_config(path)
    return path, validated


def _run_complete(experiment, config, checkpoint_root):
    path = checkpoint_root / experiment["run_name"] / "last.pt"
    if not path.is_file():
        return False, False
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_loss = experiment["loss"]
    expected_training = build_training_protocol(
        config["training"],
        build_augmentation_protocol(config["preprocessing"]),
        expected_loss,
        amp_enabled=True,
    )
    if checkpoint.get("model_name") != experiment["model_name"]:
        raise ValueError(f"Checkpoint model mismatch: {path}")
    if checkpoint.get("loss_protocol") != expected_loss:
        raise ValueError(f"Checkpoint loss protocol mismatch: {path}")
    if checkpoint.get("training_protocol") != expected_training:
        raise ValueError(f"Checkpoint training protocol mismatch: {path}")
    complete = (
        int(checkpoint["epoch"]) + 1 >= int(config["training"]["epochs"])
        or int(checkpoint["bad_epochs"]) >= int(config["training"]["patience"])
    )
    return complete, not complete


def _start_telemetry(path):
    handle = None
    try:
        path = Path(path)
        write_header = not path.is_file() or path.stat().st_size == 0
        handle = path.open("a", encoding="utf-8", newline="")
        if write_header:
            handle.write("timestamp,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,power_draw_watts\n")
        handle.flush()
        process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
                "--format=csv,noheader,nounits", "--loop-ms=5000",
            ],
            stdout=handle,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return process, handle
    except Exception as error:
        if handle is not None:
            handle.close()
        print(f"Warning: GPU telemetry could not start: {error}", flush=True)
        return None, None


def _stop_telemetry(process, handle):
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    if handle is not None:
        handle.close()


def validate_export_members(members):
    forbidden = (".pt", ".pth", ".ckpt", ".npz", ".zip")
    if any(name.lower().endswith(forbidden) for name in members):
        raise RuntimeError("Stage 2 export contains a forbidden binary artifact")
    if any("external" in name.lower() or "consensus_test" in name.lower() for name in members):
        raise RuntimeError("Stage 2 export contains an external-test artifact")


def _export_results(run_log_root, export_path):
    staging = Path(tempfile.mkdtemp(prefix="stage2-export-", dir="/content"))
    try:
        destination = staging / "outputs" / EXPERIMENT_GROUP / NOTEBOOK_GROUP / STAGE_ID / run_log_root.name
        shutil.copytree(run_log_root, destination)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if export_path.exists():
            export_path.unlink()
        shutil.make_archive(str(export_path.with_suffix("")), "zip", root_dir=staging)
        with zipfile.ZipFile(export_path) as archive:
            members = archive.namelist()
        validate_export_members(members)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run_stage2_colab(spec, repo_dir, commit_sha, drive_root=Path("/content/drive/MyDrive/DATH_CXR")):
    """Run one model's four Stage 2 experiments in an isolated Colab session."""
    import torchvision

    if spec.get("run_final_test", False):
        raise RuntimeError("External test is locked in Stage 2")
    repo_dir = Path(repo_dir)
    drive_root = Path(drive_root)
    train_archive = drive_root / "datasets" / "consensus_v1_224.zip"
    if not train_archive.is_file():
        raise FileNotFoundError(f"Missing train archive: {train_archive}")
    protocol_root = repo_dir / "data" / "protocols" / DATA_PROTOCOL_ID
    processed_root = Path("/content/datasets/consensus_v1_224")
    runtime_root = Path("/content/runtime_configs") / STAGE_ID / spec["model_slug"]
    stage_checkpoint_root = drive_root / "checkpoints" / EXPERIMENT_GROUP / NOTEBOOK_GROUP / STAGE_ID
    stage_run_root = drive_root / "run_logs" / EXPERIMENT_GROUP / NOTEBOOK_GROUP / STAGE_ID
    checkpoint_root = stage_checkpoint_root / spec["model_slug"]
    run_log_root = stage_run_root / spec["model_slug"]
    runs_root = run_log_root / "runs"
    analysis_root = run_log_root / "analysis"
    hardware_root = run_log_root / "hardware"
    for path in (checkpoint_root, runs_root, analysis_root, hardware_root):
        path.mkdir(parents=True, exist_ok=True)

    profile = _runtime_profile()
    _validate_runtime(profile)
    profile.update({
        "platform": platform.platform(),
        "torchvision": torchvision.__version__,
        "notebook_id": spec["notebook_id"],
    })
    (hardware_root / "hardware_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    if not processed_root.exists():
        temporary = Path(tempfile.mkdtemp(prefix="stage2-data-", dir="/content"))
        try:
            _safe_extract(train_archive, temporary)
            extracted = temporary / "consensus_v1_224"
            _validate_train_data(extracted, protocol_root)
            processed_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted), str(processed_root))
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    data_summary = _validate_train_data(processed_root, protocol_root)

    experiments = build_stage2_experiments(
        spec["model_name"], spec["model_slug"], spec["config"],
        spec["weighting"], spec["positive_weight"],
    )
    runtime_configs = {}
    normalized = None
    for experiment in experiments:
        path, config = _build_runtime_config(
            experiment, repo_dir, processed_root, protocol_root, runtime_root
        )
        comparable = copy.deepcopy(config)
        comparable["training"]["loss"] = "<stage2-loss-variant>"
        if normalized is not None and comparable != normalized:
            raise ValueError("Runtime configs differ beyond the Stage 2 loss variant")
        normalized = comparable
        runtime_configs[experiment["run_name"]] = (path, config)

    augmentation_protocol = build_augmentation_protocol(normalized["preprocessing"])
    contract = {
        "stage_id": STAGE_ID,
        "stage1_runtime_contract": STAGE1_RUNTIME_CONTRACT,
        "stage2_git_commit_sha": commit_sha,
        "torchvision": torchvision.__version__,
        "pip_freeze_sha256": _pip_freeze_sha256(),
        **data_summary,
        "augmentation_protocol": augmentation_protocol,
        "selection_metric": "macro_average_precision",
    }
    created = create_or_validate_contract(stage_run_root / "stage2_contract.json", contract)
    print(f"Shared Stage 2 contract: {'created' if created else 'verified'}", flush=True)

    (run_log_root / "experiment_registry.json").write_text(
        json.dumps(experiments, indent=2), encoding="utf-8"
    )
    max_new_runs = spec.get("max_new_runs_per_session")
    new_runs = 0
    for index, experiment in enumerate(experiments, start=1):
        config_path, config = runtime_configs[experiment["run_name"]]
        checkpoint_dir = checkpoint_root / experiment["run_name"]
        result_dir = runs_root / experiment["run_name"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        complete, resume = _run_complete(experiment, config, checkpoint_root)
        if complete:
            print(f"[{index}/4] skip complete: {experiment['run_name']}", flush=True)
        else:
            if max_new_runs is not None and new_runs >= max_new_runs:
                break
            command = [
                sys.executable, str(repo_dir / "scripts" / "train.py"),
                "--config", str(config_path),
                "--checkpoint-dir", str(checkpoint_dir),
                "--result-dir", str(result_dir),
            ]
            if resume:
                command.extend(["--resume", str(checkpoint_dir / "last.pt")])
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repo_dir)
            telemetry_process, telemetry_handle = _start_telemetry(
                hardware_root / f"gpu_telemetry_{experiment['run_name']}.csv"
            )
            try:
                subprocess.run(command, cwd=repo_dir, env=environment, check=True)
            finally:
                _stop_telemetry(telemetry_process, telemetry_handle)
            new_runs += 1
        analysis = analyze_stage2(experiments, run_log_root, checkpoint_root, analysis_root)
        if not analysis["macro"].empty:
            print(analysis["macro"].to_string(index=False), flush=True)

    analysis = analyze_stage2(experiments, run_log_root, checkpoint_root, analysis_root)
    performance_columns = [
        "run_name", "model", "variant", "train_hours", "mean_train_images_per_second",
        "mean_gpu_utilization", "peak_gpu_memory_allocated_gb", "peak_gpu_memory_reserved_gb",
    ]
    if not analysis["macro"].empty:
        analysis["macro"][performance_columns].to_csv(
            hardware_root / "stage2_performance_summary.csv", index=False
        )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit_sha,
        "notebook_id": spec["notebook_id"],
        "model": spec["model_name"],
        "stage_id": STAGE_ID,
        **data_summary,
        "selection_metric": "macro_average_precision",
        "external_test_used": False,
        "checkpoint_included_in_export": False,
    }
    (run_log_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    export_path = Path(spec["export_path"])
    _export_results(run_log_root, export_path)
    print(f"Stage 2 export ready: {export_path}", flush=True)
    return analysis
