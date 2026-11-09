import copy
from collections import deque
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
import time
import uuid
import zipfile

import torch

from scripts.train import build_augmentation_protocol, build_training_protocol
from src.analysis.stage2 import analyze_stage2
from src.experiments.stage2_colab import (
    DATA_PROTOCOL_ID,
    EXPERIMENT_GROUP,
    NOTEBOOK_GROUP,
    STAGE1_RUNTIME_CONTRACT,
    _build_runtime_config,
    _pip_freeze_sha256,
    _runtime_profile,
    _safe_extract,
    _start_telemetry,
    _stop_telemetry,
    _validate_runtime,
    _validate_train_data,
    validate_export_members,
)


STAGE_ID = "stage2_focalloss_v2"
FOCAL_STABILITY_EPSILON = 1e-6
MODEL_SPECS = {
    "densenet121": {
        "model_name": "densenet121",
        "config": "configs/densenet121_aug_auprc.yaml",
        "weighting": "sqrtcap3",
        "positive_weight": {
            "mode": "sqrt_ratio_capped",
            "max_value": 3.0,
        },
    },
    "resnet50": {
        "model_name": "resnet50",
        "config": "configs/resnet50_aug_auprc.yaml",
        "weighting": "raw",
        "positive_weight": {"mode": "raw_ratio", "max_value": None},
    },
    "vit_b_16": {
        "model_name": "vit_b_16",
        "config": "configs/vit_aug_auprc.yaml",
        "weighting": "sqrtcap3",
        "positive_weight": {
            "mode": "sqrt_ratio_capped",
            "max_value": 3.0,
        },
    },
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json_with_retry(path, attempts=6, delay_seconds=1.0):
    path = Path(path)
    last_error = None
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(delay_seconds * (2 ** attempt))
    raise RuntimeError(f"Unable to read verified JSON from {path}") from last_error


def write_text_with_retry(path, content, attempts=6, delay_seconds=1.0):
    path = Path(path)
    last_error = None
    for attempt in range(attempts):
        temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
            if path.read_text(encoding="utf-8") != content:
                raise OSError(f"Read-back verification failed for {path}")
            return path
        except OSError as error:
            last_error = error
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 < attempts:
                time.sleep(delay_seconds * (2 ** attempt))
    raise RuntimeError(f"Unable to write verified text to {path}") from last_error


def write_json_with_retry(path, payload, attempts=6, delay_seconds=1.0):
    path = Path(path)
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    write_text_with_retry(path, content, attempts, delay_seconds)
    actual = _read_json_with_retry(path, attempts, delay_seconds)
    if _canonical_json(actual) != _canonical_json(payload):
        raise RuntimeError(f"JSON read-back mismatch: {path}")
    return path


def create_or_validate_contract(
    path,
    expected,
    allow_create=True,
    attempts=6,
    delay_seconds=1.0,
):
    path = Path(path)
    payload = json.dumps(expected, indent=2, ensure_ascii=False)
    created = False
    last_error = None
    for attempt in range(attempts):
        try:
            if allow_create:
                try:
                    with path.open("x", encoding="utf-8") as handle:
                        handle.write(payload)
                    created = True
                except FileExistsError:
                    pass
            elif not path.is_file():
                raise FileNotFoundError(
                    "Stage 2 v2 initializer has not created the contract"
                )
            actual = _read_json_with_retry(
                path,
                attempts=max(1, attempts - attempt),
                delay_seconds=delay_seconds,
            )
            if _canonical_json(actual) != _canonical_json(expected):
                raise ValueError(f"Stage 2 v2 contract mismatch: {path}")
            return created
        except ValueError:
            raise
        except (FileNotFoundError, OSError, RuntimeError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(delay_seconds * (2 ** attempt))
    raise RuntimeError(f"Unable to create or verify contract: {path}") from last_error


def build_stage2_experiments(
    model_name,
    model_slug,
    config_path,
    weighting,
    positive_weight,
):
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
            loss.update({
                "gamma": gamma,
                "stability_epsilon": FOCAL_STABILITY_EPSILON,
            })
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
                f"{model_slug}_stage2v2_{run_loss}_{weighting}_"
                "aug_auprc_s8515_exttest_v1_seed42_amp"
            ),
        })
    if len({item["run_name"] for item in experiments}) != 4:
        raise ValueError("Stage 2 v2 run names must be unique")
    return experiments


def build_all_stage2_experiments():
    return {
        slug: build_stage2_experiments(
            item["model_name"],
            slug,
            item["config"],
            item["weighting"],
            item["positive_weight"],
        )
        for slug, item in MODEL_SPECS.items()
    }


def build_experiment_protocol():
    all_experiments = build_all_stage2_experiments()
    return {
        "stage_id": STAGE_ID,
        "variants": ["bce", "focal_g05", "focal_g10", "focal_g15"],
        "focal_math_dtype": "float32",
        "focal_stability_epsilon": FOCAL_STABILITY_EPSILON,
        "models": {
            slug: {
                "model_name": MODEL_SPECS[slug]["model_name"],
                "weighting": MODEL_SPECS[slug]["weighting"],
                "positive_weight": MODEL_SPECS[slug]["positive_weight"],
                "runs": [
                    {
                        "variant": item["variant"],
                        "loss": item["loss"],
                        "run_name": item["run_name"],
                    }
                    for item in all_experiments[slug]
                ],
            }
            for slug in MODEL_SPECS
        },
    }


def _validate_spec(spec):
    if spec.get("bootstrap_role") not in {"initializer", "follower"}:
        raise ValueError("bootstrap_role must be initializer or follower")
    slug = spec.get("model_slug")
    if slug not in MODEL_SPECS:
        raise ValueError(f"Unsupported Stage 2 v2 model slug: {slug}")
    expected = MODEL_SPECS[slug]
    for key in ("model_name", "config", "weighting", "positive_weight"):
        if spec.get(key) != expected[key]:
            raise ValueError(f"Stage 2 v2 spec mismatch for {slug}/{key}")
    if spec.get("stage_id") != STAGE_ID:
        raise ValueError(f"Stage 2 v2 requires stage_id={STAGE_ID}")


def _mkdir_with_retry(path, attempts=6, delay_seconds=1.0):
    path = Path(path)
    last_error = None
    for attempt in range(attempts):
        try:
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir():
                raise FileNotFoundError(path)
            return path
        except OSError as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(delay_seconds * (2 ** attempt))
    raise RuntimeError(f"Unable to create Drive directory: {path}") from last_error


def _duplicate_stage_paths(parent):
    parent = Path(parent)
    if not parent.is_dir():
        return []
    return [
        item
        for item in parent.iterdir()
        if item.name == STAGE_ID or item.name.startswith(f"{STAGE_ID} (")
    ]


def _assert_unique_stage(parent):
    candidates = _duplicate_stage_paths(parent)
    exact = [item for item in candidates if item.name == STAGE_ID]
    if len(candidates) != 1 or len(exact) != 1:
        raise RuntimeError(
            f"Expected one canonical {STAGE_ID} under {parent}; "
            f"found {[item.name for item in candidates]}"
        )


def _wait_for_paths(paths, timeout_seconds, poll_seconds=5.0):
    paths = [Path(path) for path in paths]
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if all(path.exists() for path in paths):
            return
        time.sleep(poll_seconds)
    missing = [str(path) for path in paths if not path.exists()]
    raise TimeoutError(
        "Stage 2 v2 initializer was not ready before timeout. Missing: "
        + ", ".join(missing)
    )


def drive_write_probe(path):
    path = Path(path)
    token = uuid.uuid4().hex
    probe = path / f".stage2v2-write-probe-{token}.txt"
    write_text_with_retry(probe, token)
    if probe.read_text(encoding="utf-8") != token:
        raise RuntimeError(f"Drive write probe failed: {path}")
    last_error = None
    for attempt in range(6):
        try:
            probe.unlink(missing_ok=True)
            if not probe.exists():
                break
        except OSError as error:
            last_error = error
        if attempt < 5:
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Drive probe cleanup failed: {probe}") from last_error


def _bootstrap_drive(
    role,
    stage_checkpoint_root,
    stage_run_root,
    expected_contract,
    current_model_slug,
    timeout_seconds,
):
    checkpoint_parent = stage_checkpoint_root.parent
    run_parent = stage_run_root.parent
    all_experiments = build_all_stage2_experiments()
    contract_path = stage_run_root / "stage2_contract.json"

    if role == "initializer":
        _mkdir_with_retry(checkpoint_parent)
        _mkdir_with_retry(run_parent)
        existing_checkpoint = _duplicate_stage_paths(checkpoint_parent)
        existing_run = _duplicate_stage_paths(run_parent)
        invalid_existing = any(
            candidates
            and (
                len(candidates) != 1
                or candidates[0].name != STAGE_ID
            )
            for candidates in (existing_checkpoint, existing_run)
        )
        if invalid_existing:
            raise RuntimeError("Duplicate Stage 2 v2 Drive roots detected")
        _mkdir_with_retry(stage_checkpoint_root)
        _mkdir_with_retry(stage_run_root)
        for slug, experiments in all_experiments.items():
            checkpoint_model = _mkdir_with_retry(stage_checkpoint_root / slug)
            run_model = _mkdir_with_retry(stage_run_root / slug)
            runs_root = _mkdir_with_retry(run_model / "runs")
            _mkdir_with_retry(run_model / "analysis")
            _mkdir_with_retry(run_model / "hardware")
            for experiment in experiments:
                _mkdir_with_retry(checkpoint_model / experiment["run_name"])
                result_dir = _mkdir_with_retry(
                    runs_root / experiment["run_name"]
                )
                status_path = result_dir / "run_status.json"
                if not status_path.exists():
                    write_json_with_retry(status_path, {
                        "state": "pending",
                        "updated_at_utc": _utc_now(),
                        "run_name": experiment["run_name"],
                    })
        create_or_validate_contract(
            contract_path,
            expected_contract,
            allow_create=True,
        )
        _assert_unique_stage(checkpoint_parent)
        _assert_unique_stage(run_parent)
        for slug in MODEL_SPECS:
            drive_write_probe(stage_checkpoint_root / slug)
            drive_write_probe(stage_run_root / slug / "runs")
            drive_write_probe(stage_run_root / slug / "analysis")
            drive_write_probe(stage_run_root / slug / "hardware")
        print("STAGE 2 V2 BOOTSTRAP READY", flush=True)
        return

    required = [contract_path]
    for slug, experiments in all_experiments.items():
        required.extend([
            stage_checkpoint_root / slug,
            stage_run_root / slug / "runs",
            stage_run_root / slug / "analysis",
            stage_run_root / slug / "hardware",
        ])
        for experiment in experiments:
            required.extend([
                stage_checkpoint_root / slug / experiment["run_name"],
                stage_run_root / slug / "runs" / experiment["run_name"],
            ])
    _wait_for_paths(required, timeout_seconds)
    _assert_unique_stage(checkpoint_parent)
    _assert_unique_stage(run_parent)
    create_or_validate_contract(
        contract_path,
        expected_contract,
        allow_create=False,
    )
    drive_write_probe(stage_checkpoint_root / current_model_slug)
    drive_write_probe(stage_run_root / current_model_slug / "runs")
    drive_write_probe(stage_run_root / current_model_slug / "analysis")
    drive_write_probe(stage_run_root / current_model_slug / "hardware")
    print("STAGE 2 V2 FOLLOWER VERIFIED", flush=True)


def _nonfinite_tensor_paths(value, prefix):
    if torch.is_tensor(value):
        if value.is_floating_point() and not torch.isfinite(value).all():
            return [prefix]
        return []
    if isinstance(value, dict):
        paths = []
        for key, item in value.items():
            paths.extend(_nonfinite_tensor_paths(item, f"{prefix}.{key}"))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_nonfinite_tensor_paths(item, f"{prefix}[{index}]"))
        return paths
    return []


def _checkpoint_is_finite(checkpoint, path):
    bad_paths = []
    for field in (
        "model_state_dict",
        "optimizer_state_dict",
        "amp_scaler_state_dict",
        "pos_weight",
        "raw_pos_weight",
    ):
        if field in checkpoint:
            bad_paths.extend(_nonfinite_tensor_paths(checkpoint[field], field))
    if bad_paths:
        raise ValueError(
            f"Checkpoint contains nonfinite tensors {bad_paths[:10]}: {path}"
        )


def _run_completion(
    experiment,
    config,
    checkpoint_root,
    amp_enabled=True,
):
    path = checkpoint_root / experiment["run_name"] / "last.pt"
    if not path.is_file():
        return False, False, None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _checkpoint_is_finite(checkpoint, path)
    expected_loss = experiment["loss"]
    expected_training = build_training_protocol(
        config["training"],
        build_augmentation_protocol(config["preprocessing"]),
        expected_loss,
        amp_enabled=bool(amp_enabled),
    )
    if checkpoint.get("model_name") != experiment["model_name"]:
        raise ValueError(f"Checkpoint model mismatch: {path}")
    if checkpoint.get("loss_protocol") != expected_loss:
        raise ValueError(f"Checkpoint loss protocol mismatch: {path}")
    if checkpoint.get("training_protocol") != expected_training:
        raise ValueError(f"Checkpoint training protocol mismatch: {path}")
    epoch_limit = int(checkpoint["epoch"]) + 1 >= int(
        config["training"]["epochs"]
    )
    early_stopped = int(checkpoint["bad_epochs"]) >= int(
        config["training"]["patience"]
    )
    complete = epoch_limit or early_stopped
    stop_reason = (
        "epoch_limit" if epoch_limit else "early_stopping" if early_stopped else None
    )
    return complete, not complete, stop_reason


def _last_safe_epoch(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        return None
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        return int(checkpoint["epoch"]) + 1
    except Exception:
        return None


def _run_subprocess_streaming(
    command,
    cwd,
    environment,
    result_dir,
    local_root=None,
):
    local_root = Path(local_root or "/content/stage2v2_subprocess_logs")
    local_root.mkdir(parents=True, exist_ok=True)
    local_log = local_root / f"{result_dir.name}.log"
    tail = deque(maxlen=200)
    with local_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
            tail.append(line.rstrip("\n"))
        return_code = process.wait()
    write_text_with_retry(
        result_dir / "train_subprocess.log",
        local_log.read_text(encoding="utf-8"),
    )
    return return_code, "\n".join(tail)


def _export_results(run_log_root, export_path):
    staging = Path(tempfile.mkdtemp(prefix="stage2v2-export-", dir="/content"))
    try:
        destination = (
            staging
            / "outputs"
            / EXPERIMENT_GROUP
            / NOTEBOOK_GROUP
            / STAGE_ID
            / run_log_root.name
        )
        shutil.copytree(run_log_root, destination)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if export_path.exists():
            export_path.unlink()
        shutil.make_archive(
            str(export_path.with_suffix("")),
            "zip",
            root_dir=staging,
        )
        with zipfile.ZipFile(export_path) as archive:
            validate_export_members(archive.namelist())
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prepare_stage2_colab(
    spec,
    repo_dir,
    commit_sha,
    drive_root=Path("/content/drive/MyDrive/DATH_CXR"),
):
    """Validate runtime/data and bootstrap or verify Stage 2 v2 Drive state."""
    import torchvision

    _validate_spec(spec)
    if spec.get("run_final_test", False):
        raise RuntimeError("External test is locked in Stage 2 v2")
    repo_dir = Path(repo_dir)
    drive_root = Path(drive_root)
    train_archive = drive_root / "datasets" / "consensus_v1_224.zip"
    if not train_archive.is_file():
        raise FileNotFoundError(f"Missing train archive: {train_archive}")
    protocol_root = repo_dir / "data" / "protocols" / DATA_PROTOCOL_ID
    processed_root = Path("/content/datasets/consensus_v1_224")
    runtime_root = Path("/content/runtime_configs") / STAGE_ID / spec["model_slug"]
    stage_checkpoint_root = (
        drive_root / "checkpoints" / EXPERIMENT_GROUP / NOTEBOOK_GROUP / STAGE_ID
    )
    stage_run_root = (
        drive_root / "run_logs" / EXPERIMENT_GROUP / NOTEBOOK_GROUP / STAGE_ID
    )

    profile = _runtime_profile()
    _validate_runtime(profile)
    profile.update({
        "platform": platform.platform(),
        "torchvision": torchvision.__version__,
        "notebook_id": spec["notebook_id"],
    })

    if not processed_root.exists():
        temporary = Path(tempfile.mkdtemp(prefix="stage2v2-data-", dir="/content"))
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
        spec["model_name"],
        spec["model_slug"],
        spec["config"],
        spec["weighting"],
        spec["positive_weight"],
    )
    runtime_configs = {}
    normalized = None
    for experiment in experiments:
        path, config = _build_runtime_config(
            experiment,
            repo_dir,
            processed_root,
            protocol_root,
            runtime_root,
        )
        comparable = copy.deepcopy(config)
        comparable["training"]["loss"] = "<stage2v2-loss-variant>"
        if normalized is not None and comparable != normalized:
            raise ValueError("Runtime configs differ beyond the loss variant")
        normalized = comparable
        runtime_configs[experiment["run_name"]] = (path, config)

    experiment_protocol = build_experiment_protocol()
    contract = {
        "stage_id": STAGE_ID,
        "stage1_runtime_contract": STAGE1_RUNTIME_CONTRACT,
        "stage2_git_commit_sha": commit_sha,
        "torchvision": torchvision.__version__,
        "pip_freeze_sha256": _pip_freeze_sha256(),
        **data_summary,
        "augmentation_protocol": build_augmentation_protocol(
            normalized["preprocessing"]
        ),
        "selection_metric": "macro_average_precision",
        "experiment_protocol": experiment_protocol,
        "experiment_protocol_sha256": _sha256_text(
            _canonical_json(experiment_protocol)
        ),
    }
    _bootstrap_drive(
        spec["bootstrap_role"],
        stage_checkpoint_root,
        stage_run_root,
        contract,
        spec["model_slug"],
        float(spec.get("bootstrap_timeout_seconds", 600)),
    )

    checkpoint_root = stage_checkpoint_root / spec["model_slug"]
    run_log_root = stage_run_root / spec["model_slug"]
    hardware_root = run_log_root / "hardware"
    write_json_with_retry(hardware_root / "hardware_profile.json", profile)
    write_json_with_retry(
        run_log_root / "experiment_registry.json",
        experiments,
    )
    print(
        f"Shared Stage 2 v2 contract verified for {spec['model_slug']}",
        flush=True,
    )
    return {
        "spec": copy.deepcopy(spec),
        "repo_dir": repo_dir,
        "commit_sha": commit_sha,
        "experiments": experiments,
        "runtime_configs": runtime_configs,
        "data_summary": data_summary,
        "checkpoint_root": checkpoint_root,
        "run_log_root": run_log_root,
        "runs_root": run_log_root / "runs",
        "analysis_root": run_log_root / "analysis",
        "hardware_root": hardware_root,
    }


def run_stage2_colab(
    spec,
    repo_dir,
    commit_sha,
    drive_root=Path("/content/drive/MyDrive/DATH_CXR"),
    prepared_context=None,
):
    """Train or resume one model after Stage 2 v2 preparation succeeds."""
    context = prepared_context or prepare_stage2_colab(
        spec,
        repo_dir,
        commit_sha,
        drive_root,
    )
    if context["spec"] != spec or context["commit_sha"] != commit_sha:
        raise ValueError("Prepared Stage 2 v2 context mismatch")
    experiments = context["experiments"]
    runtime_configs = context["runtime_configs"]
    checkpoint_root = context["checkpoint_root"]
    run_log_root = context["run_log_root"]
    runs_root = context["runs_root"]
    analysis_root = context["analysis_root"]
    hardware_root = context["hardware_root"]
    repo_dir = context["repo_dir"]
    max_new_runs = spec.get("max_new_runs_per_session")
    new_runs = 0

    for index, experiment in enumerate(experiments, start=1):
        config_path, config = runtime_configs[experiment["run_name"]]
        checkpoint_dir = checkpoint_root / experiment["run_name"]
        result_dir = runs_root / experiment["run_name"]
        if not checkpoint_dir.is_dir() or not result_dir.is_dir():
            raise FileNotFoundError(
                "Initializer did not create expected run directories: "
                f"{checkpoint_dir} / {result_dir}"
            )
        complete, resume, stop_reason = _run_completion(
            experiment,
            config,
            checkpoint_root,
        )
        status_path = result_dir / "run_status.json"
        if complete:
            write_json_with_retry(status_path, {
                "state": "complete",
                "updated_at_utc": _utc_now(),
                "stop_reason": stop_reason,
                "run_name": experiment["run_name"],
            })
            print(f"[{index}/4] skip complete: {experiment['run_name']}")
        else:
            if max_new_runs is not None and new_runs >= max_new_runs:
                break
            state = "resuming" if resume else "running"
            write_json_with_retry(status_path, {
                "state": state,
                "updated_at_utc": _utc_now(),
                "run_name": experiment["run_name"],
                "last_safe_epoch": _last_safe_epoch(
                    checkpoint_dir / "last.pt"
                ),
            })
            command = [
                sys.executable,
                str(repo_dir / "scripts" / "train.py"),
                "--config",
                str(config_path),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--result-dir",
                str(result_dir),
            ]
            if resume:
                command.extend(["--resume", str(checkpoint_dir / "last.pt")])
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repo_dir)
            environment["PYTHONUNBUFFERED"] = "1"
            telemetry_process, telemetry_handle = _start_telemetry(
                hardware_root / f"gpu_telemetry_{experiment['run_name']}.csv"
            )
            try:
                try:
                    return_code, output_tail = _run_subprocess_streaming(
                        command,
                        repo_dir,
                        environment,
                        result_dir,
                    )
                except Exception as error:
                    return_code = -1
                    output_tail = f"{type(error).__name__}: {error}"
            finally:
                _stop_telemetry(telemetry_process, telemetry_handle)
            if return_code != 0:
                failure = {
                    "state": "failed",
                    "updated_at_utc": _utc_now(),
                    "run_name": experiment["run_name"],
                    "return_code": return_code,
                    "last_safe_epoch": _last_safe_epoch(
                        checkpoint_dir / "last.pt"
                    ),
                    "traceback_tail": output_tail,
                    "log_file": "train_subprocess.log",
                }
                write_json_with_retry(
                    result_dir / "training_failure.json",
                    failure,
                )
                write_json_with_retry(status_path, failure)
                analyze_stage2(
                    experiments,
                    run_log_root,
                    checkpoint_root,
                    analysis_root,
                )
                raise RuntimeError(
                    "Training subprocess failed for "
                    f"{experiment['run_name']}. Full log: "
                    f"{result_dir / 'train_subprocess.log'}\n{output_tail}"
                )
            complete, _, stop_reason = _run_completion(
                experiment,
                config,
                checkpoint_root,
            )
            if not complete:
                raise RuntimeError(
                    "Training subprocess exited successfully without a complete run"
                )
            write_json_with_retry(status_path, {
                "state": "complete",
                "updated_at_utc": _utc_now(),
                "stop_reason": stop_reason,
                "run_name": experiment["run_name"],
            })
            (result_dir / "training_failure.json").unlink(missing_ok=True)
            new_runs += 1
        analysis = analyze_stage2(
            experiments,
            run_log_root,
            checkpoint_root,
            analysis_root,
        )
        if not analysis["macro"].empty:
            print(analysis["macro"].to_string(index=False), flush=True)

    analysis = analyze_stage2(
        experiments,
        run_log_root,
        checkpoint_root,
        analysis_root,
    )
    performance_columns = [
        "run_name",
        "model",
        "variant",
        "train_hours",
        "mean_train_images_per_second",
        "mean_gpu_utilization",
        "peak_gpu_memory_allocated_gb",
        "peak_gpu_memory_reserved_gb",
    ]
    if not analysis["macro"].empty:
        analysis["macro"][performance_columns].to_csv(
            hardware_root / "stage2_performance_summary.csv",
            index=False,
        )
    manifest = {
        "created_at_utc": _utc_now(),
        "git_commit": commit_sha,
        "notebook_id": spec["notebook_id"],
        "model": spec["model_name"],
        "stage_id": STAGE_ID,
        **context["data_summary"],
        "selection_metric": "macro_average_precision",
        "external_test_used": False,
        "checkpoint_included_in_export": False,
    }
    write_json_with_retry(
        run_log_root / "experiment_manifest.json",
        manifest,
    )
    export_path = Path(spec["export_path"])
    _export_results(run_log_root, export_path)
    print(f"Stage 2 v2 export ready: {export_path}", flush=True)
    return analysis
