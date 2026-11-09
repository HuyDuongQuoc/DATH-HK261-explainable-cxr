import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.train import build_training_protocol, validate_resume_training_protocol
from src.analysis.stage2 import analyze_stage2
import src.experiments.stage2_colab_v2 as stage2_v2
from src.experiments.stage2_colab_v2 import (
    FOCAL_STABILITY_EPSILON,
    STAGE_ID,
    STAGE1_RUNTIME_CONTRACT,
    _assert_unique_stage,
    _bootstrap_drive,
    _checkpoint_is_finite,
    _run_subprocess_streaming,
    build_stage2_experiments,
    create_or_validate_contract,
    validate_export_members,
    write_json_with_retry,
)
from src.losses.classification_loss import (
    SigmoidFocalLoss,
    build_classification_loss,
    build_loss_from_checkpoint,
    validate_checkpoint_loss_protocol,
)
from src.training.engine import run_epoch
from src.utils.config import validate_loss_config


def _focal_protocol(gamma=1.0, mode="sqrt_ratio_capped", maximum=3.0):
    return {
        "name": "sigmoid_focal",
        "gamma": gamma,
        "stability_epsilon": FOCAL_STABILITY_EPSILON,
        "positive_weight": {"mode": mode, "max_value": maximum},
    }


def test_focal_gamma_zero_matches_weighted_bce():
    logits = torch.tensor([[1.2, -0.7], [-0.1, 2.1]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pos_weight = torch.tensor([3.0, 1.5])
    focal = SigmoidFocalLoss(
        pos_weight,
        gamma=0.0,
        stability_epsilon=FOCAL_STABILITY_EPSILON,
    )(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    assert torch.allclose(focal, bce, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5])
def test_focal_forward_backward_is_finite(gamma):
    logits = torch.randn(8, 14, requires_grad=True)
    targets = torch.randint(0, 2, (8, 14), dtype=torch.float32)
    loss = SigmoidFocalLoss(
        torch.linspace(1, 5, 14),
        gamma,
        FOCAL_STABILITY_EPSILON,
    )(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is unavailable")
@pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5])
def test_focal_amp_is_finite(gamma):
    logits = torch.randn(8, 14, device="cuda", requires_grad=True)
    targets = torch.randint(0, 2, (8, 14), device="cuda", dtype=torch.float32)
    criterion = SigmoidFocalLoss(
        torch.ones(14, device="cuda"),
        gamma,
        FOCAL_STABILITY_EPSILON,
    ).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is unavailable")
@pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5])
def test_focal_extreme_logits_remain_finite_across_cuda_amp_steps(gamma):
    logits = torch.nn.Parameter(torch.tensor(
        [[50.0, -50.0, 30.0, -30.0, 20.0, -20.0]],
        device="cuda",
    ))
    targets = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]],
        device="cuda",
    )
    criterion = SigmoidFocalLoss(
        torch.ones(6, device="cuda"),
        gamma,
        FOCAL_STABILITY_EPSILON,
    ).cuda()
    optimizer = torch.optim.AdamW([logits], lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()
        scaler.step(optimizer)
        scaler.update()
        assert torch.isfinite(logits).all()


@pytest.mark.parametrize("gamma", [None, -1.0, 0.0, float("nan"), 5.1])
def test_focal_config_rejects_invalid_gamma(gamma):
    training = {
        "loss": {
            "name": "sigmoid_focal",
            "gamma": gamma,
            "stability_epsilon": FOCAL_STABILITY_EPSILON,
            "positive_weight": {"mode": "raw_ratio", "max_value": None},
        }
    }
    with pytest.raises(ValueError, match="gamma"):
        validate_loss_config(training)


@pytest.mark.parametrize(
    "epsilon",
    [None, -1.0, 0.0, 1.0, float("nan")],
)
def test_focal_config_rejects_invalid_stability_epsilon(epsilon):
    training = {
        "loss": {
            "name": "sigmoid_focal",
            "gamma": 0.5,
            "stability_epsilon": epsilon,
            "positive_weight": {"mode": "raw_ratio", "max_value": None},
        }
    }
    with pytest.raises(ValueError, match="stability_epsilon"):
        validate_loss_config(training)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5])
def test_focal_extreme_logits_remain_finite_across_optimizer_steps(gamma):
    logits = torch.nn.Parameter(torch.tensor([
        [50.0, -50.0, 30.0, -30.0, 20.0, -20.0],
    ]))
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    criterion = SigmoidFocalLoss(
        torch.ones(6),
        gamma,
        FOCAL_STABILITY_EPSILON,
    )
    optimizer = torch.optim.AdamW([logits], lr=1e-3)
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(logits, targets)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()
        optimizer.step()
        assert torch.isfinite(logits).all()


def test_engine_rejects_nonfinite_gradient_before_optimizer_step():
    class InfiniteGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits):
            ctx.save_for_backward(logits)
            return logits.new_tensor(1.0)

        @staticmethod
        def backward(ctx, grad_output):
            (logits,) = ctx.saved_tensors
            return torch.full_like(logits, float("inf"))

    class Criterion(torch.nn.Module):
        def forward(self, logits, targets):
            return InfiniteGradient.apply(logits)

    samples = [
        {
            "image": torch.ones(3, 2, 2),
            "labels": torch.tensor([1.0]),
            "image_id": "sample",
        }
    ]
    loader = DataLoader(samples, batch_size=1)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    with pytest.raises(RuntimeError, match="Nonfinite gradient.*batch 0"):
        run_epoch(
            model,
            loader,
            Criterion(),
            torch.device("cpu"),
            optimizer=optimizer,
            collect_predictions=False,
            description="gradient-guard-test",
        )
    assert all(
        torch.equal(old, current)
        for old, current in zip(before, model.parameters())
    )


def test_weighted_bce_rejects_gamma_and_focal_uses_effective_weight():
    for gamma in (None, 1.0):
        with pytest.raises(ValueError, match="gamma"):
            validate_loss_config({
                "loss": {
                    "name": "weighted_bce",
                    "gamma": gamma,
                    "positive_weight": {"mode": "raw_ratio", "max_value": None},
                }
            })
    with pytest.raises(ValueError, match="stability_epsilon"):
        validate_loss_config({
            "loss": {
                "name": "weighted_bce",
                "stability_epsilon": FOCAL_STABILITY_EPSILON,
                "positive_weight": {"mode": "raw_ratio", "max_value": None},
            }
        })
    labels = np.asarray([[1, 1], [0, 0], [0, 1], [0, 0]], dtype=float)
    criterion, pos_weight, protocol, _ = build_classification_loss(
        labels, "cpu", _focal_protocol()
    )
    assert isinstance(criterion, SigmoidFocalLoss)
    assert torch.allclose(
        pos_weight,
        torch.tensor([np.sqrt(3), 1.0], dtype=torch.float32),
    )
    assert torch.allclose(criterion.pos_weight, pos_weight)
    assert protocol == _focal_protocol()


def test_focal_checkpoint_round_trip_and_resume_mismatch():
    protocol = _focal_protocol(1.0)
    checkpoint = {"pos_weight": torch.tensor([2.0]), "loss_protocol": protocol}
    criterion, restored = build_loss_from_checkpoint(checkpoint, torch.device("cpu"))
    assert isinstance(criterion, SigmoidFocalLoss)
    assert criterion.gamma == 1.0
    assert restored == protocol
    validate_checkpoint_loss_protocol(checkpoint, protocol)
    with pytest.raises(ValueError, match="loss protocol"):
        validate_checkpoint_loss_protocol(checkpoint, _focal_protocol(1.5))

    training = {
        "batch_size": 8, "validation_batch_size": 64, "num_workers": 2,
        "persistent_workers": True, "prefetch_factor": 2,
        "monitor": "macro_average_precision",
    }
    expected = build_training_protocol(
        training, {"enabled": True, "config": {}}, protocol, amp_enabled=True
    )
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        validate_resume_training_protocol({}, expected)


def test_shared_contract_exclusive_create_and_mismatch(tmp_path):
    path = tmp_path / "stage2_contract.json"
    contract = {"stage1_runtime_contract": STAGE1_RUNTIME_CONTRACT, "commit": "abc"}
    assert create_or_validate_contract(path, contract) is True
    assert create_or_validate_contract(path, contract) is False
    with pytest.raises(ValueError, match="contract mismatch"):
        create_or_validate_contract(path, {**contract, "commit": "different"})


def test_initializer_creates_all_subtrees_and_follower_never_creates(tmp_path):
    checkpoint_root = tmp_path / "checkpoints" / STAGE_ID
    run_root = tmp_path / "run_logs" / STAGE_ID
    contract = {"stage_id": STAGE_ID, "commit": "abc"}
    _bootstrap_drive(
        "initializer",
        checkpoint_root,
        run_root,
        contract,
        "densenet121",
        timeout_seconds=0,
    )
    assert (run_root / "stage2_contract.json").is_file()
    for slug in ("densenet121", "resnet50", "vit_b_16"):
        assert (checkpoint_root / slug).is_dir()
        assert (run_root / slug / "runs").is_dir()
        assert (run_root / slug / "analysis").is_dir()
        assert (run_root / slug / "hardware").is_dir()
        statuses = list((run_root / slug / "runs").glob("*/run_status.json"))
        assert len(statuses) == 4
        assert all(
            json.loads(path.read_text(encoding="utf-8"))["state"] == "pending"
            for path in statuses
        )

    missing_checkpoint = tmp_path / "missing-checkpoints" / STAGE_ID
    missing_run = tmp_path / "missing-runs" / STAGE_ID
    with pytest.raises(TimeoutError, match="initializer"):
        _bootstrap_drive(
            "follower",
            missing_checkpoint,
            missing_run,
            contract,
            "resnet50",
            timeout_seconds=0,
        )
    assert not missing_checkpoint.exists()
    assert not missing_run.exists()


def test_duplicate_stage_root_is_rejected(tmp_path):
    parent = tmp_path / "run_logs"
    (parent / STAGE_ID).mkdir(parents=True)
    (parent / f"{STAGE_ID} (1)").mkdir()
    with pytest.raises(RuntimeError, match="canonical"):
        _assert_unique_stage(parent)


def test_resume_rejects_nonfinite_optimizer_checkpoint(tmp_path):
    checkpoint = {
        "model_state_dict": {"weight": torch.ones(1)},
        "optimizer_state_dict": {
            "state": {0: {"exp_avg": torch.tensor([float("inf")])}},
        },
        "pos_weight": torch.ones(1),
    }
    with pytest.raises(ValueError, match="optimizer_state_dict"):
        _checkpoint_is_finite(checkpoint, tmp_path / "last.pt")


def test_verified_json_write_retries_and_subprocess_failure_log(
    tmp_path,
    monkeypatch,
):
    json_path = tmp_path / "status.json"
    payload = {"state": "failed", "last_safe_epoch": 1}
    real_replace = os.replace
    attempts = {"count": 0}

    def transient_replace(source, destination):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("transient DriveFS rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr(stage2_v2.os, "replace", transient_replace)
    write_json_with_retry(json_path, payload, attempts=2, delay_seconds=0)
    assert attempts["count"] == 2
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload

    result_dir = tmp_path / "run"
    result_dir.mkdir()
    child_environment = dict(os.environ)
    child_environment["PYTHONIOENCODING"] = "utf-8"
    child_environment["PYTHONUTF8"] = "1"
    return_code, output_tail = _run_subprocess_streaming(
        [
            sys.executable,
            "-c",
            "import sys; print('child-traceback-marker: ĐATH'); "
            "print('lỗi UTF-8', file=sys.stderr); sys.exit(3)",
        ],
        tmp_path,
        child_environment,
        result_dir,
        local_root=tmp_path / "local-logs",
    )
    assert return_code == 3
    assert "child-traceback-marker: ĐATH" in output_tail
    assert "lỗi UTF-8" in output_tail
    assert "child-traceback-marker: ĐATH" in (
        result_dir / "train_subprocess.log"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "member",
    ["best.pt", "mask.npz", "nested/archive.zip", "external_test_metrics.json"],
)
def test_stage2_export_rejects_binary_and_external_artifacts(member):
    with pytest.raises(RuntimeError, match="artifact"):
        validate_export_members(["analysis/report.md", member])
    validate_export_members(["analysis/report.md", "runs/best_val_predictions.csv"])


def _write_stage2_run(run_root, checkpoint_root, experiment, score, recall=0.5):
    findings = ["a", "b"]
    result_dir = run_root / "runs" / experiment["run_name"]
    result_dir.mkdir(parents=True)
    checkpoint_dir = checkpoint_root / experiment["run_name"]
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pt").write_bytes(b"checkpoint")
    per_class = [
        {
            "finding": finding, "threshold": 0.5, "auroc": min(score + 0.2, 1.0),
            "average_precision": score, "precision": score, "recall": recall,
            "f1": score, "positive_count": 2, "negative_count": 2,
        }
        for finding in findings
    ]
    metrics = {
        "per_class": per_class, "macro_auroc": min(score + 0.2, 1.0),
        "macro_average_precision": score, "macro_precision": score,
        "macro_recall": recall, "macro_f1": score, "checkpoint_epoch": 2,
    }
    (result_dir / "best_val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame({
        "image_id": ["i1", "i2", "i3", "i4"],
        "target::a": [1, 0, 1, 0], "probability::a": [0.9, 0.2, 0.8, 0.1],
        "target::b": [0, 1, 0, 1], "probability::b": [0.1, 0.9, 0.2, 0.8],
    }).to_csv(result_dir / "best_val_predictions.csv", index=False)
    (result_dir / "loss_metadata.json").write_text(json.dumps({"loss_protocol": experiment["loss"]}), encoding="utf-8")
    (result_dir / "run_status.json").write_text(
        json.dumps({"state": "complete", "stop_reason": "early_stopping"}),
        encoding="utf-8",
    )
    pd.DataFrame([
        {"epoch": 1, "val_macro_ap": score - 0.004, "train_seconds": 1, "val_seconds": 1,
         "train_images_per_second": 2, "peak_gpu_memory_gb": 0,
         "peak_gpu_memory_allocated_gb": 0, "peak_gpu_memory_reserved_gb": 0},
        {"epoch": 2, "val_macro_ap": score, "train_seconds": 1, "val_seconds": 1,
         "train_images_per_second": 2, "peak_gpu_memory_gb": 0,
         "peak_gpu_memory_allocated_gb": 0, "peak_gpu_memory_reserved_gb": 0},
    ]).to_csv(result_dir / "history.csv", index=False)


def _stage2_experiments():
    return build_stage2_experiments(
        "densenet121", "densenet121", "configs/densenet121_aug_auprc.yaml",
        "sqrtcap3", {"mode": "sqrt_ratio_capped", "max_value": 3.0},
    )


def test_stage2_analysis_selects_focal_and_falls_back_to_bce(tmp_path):
    experiments = _stage2_experiments()
    checkpoint_root = tmp_path / "checkpoints"
    run_root = tmp_path / "logs"
    for experiment, score in zip(experiments, [0.50, 0.506, 0.520, 0.510]):
        _write_stage2_run(run_root, checkpoint_root, experiment, score)
    result = analyze_stage2(experiments, run_root, checkpoint_root)
    assert result["selection"]["winner"] == "focal_g10"
    assert result["selection"]["focal_replaced_bce"] is True
    assert (run_root / "analysis" / "stage2_analysis_report.md").is_file()

    fallback_root = tmp_path / "fallback"
    fallback_checkpoints = tmp_path / "fallback-checkpoints"
    for experiment, score in zip(experiments, [0.50, 0.504, 0.49, 0.48]):
        _write_stage2_run(fallback_root, fallback_checkpoints, experiment, score)
    fallback = analyze_stage2(experiments, fallback_root, fallback_checkpoints)
    assert fallback["selection"]["winner"] == "bce"
    assert fallback["selection"]["focal_replaced_bce"] is False


def test_stage2_partial_analysis_does_not_publish_selection(tmp_path):
    experiments = _stage2_experiments()
    _write_stage2_run(tmp_path / "logs", tmp_path / "checkpoints", experiments[0], 0.5)
    result = analyze_stage2(experiments, tmp_path / "logs", tmp_path / "checkpoints")
    assert result["selection"] == {}


def test_stage2_failed_partial_run_is_not_complete(tmp_path):
    experiments = _stage2_experiments()
    run_root = tmp_path / "logs"
    checkpoint_root = tmp_path / "checkpoints"
    _write_stage2_run(run_root, checkpoint_root, experiments[0], 0.5)
    result_dir = run_root / "runs" / experiments[0]["run_name"]
    (result_dir / "run_status.json").write_text(
        json.dumps({"state": "failed", "last_safe_epoch": 1}),
        encoding="utf-8",
    )
    result = analyze_stage2(experiments, run_root, checkpoint_root)
    assert result["status"].iloc[0].status == "failed"
    assert result["selection"] == {}


@pytest.mark.parametrize(
    "notebook_name,model_name,weighting,bootstrap_role",
    [
        (
            "04_1_densenet121_focalloss.ipynb",
            "densenet121",
            "sqrtcap3",
            "initializer",
        ),
        (
            "04_2_resnet50_focalloss.ipynb",
            "resnet50",
            "raw",
            "follower",
        ),
        (
            "04_3_vit_b16_focalloss.ipynb",
            "vit_b_16",
            "sqrtcap3",
            "follower",
        ),
    ],
)
def test_stage2_notebook_contract(
    notebook_name,
    model_name,
    weighting,
    bootstrap_role,
):
    notebook = json.loads((Path("notebooks") / notebook_name).read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    compile(code, notebook_name, "exec")
    assert f'"model_name": "{model_name}"' in code
    assert f'"weighting": "{weighting}"' in code
    assert f'BOOTSTRAP_ROLE = "{bootstrap_role}"' in code
    assert 'STAGE_ID = "stage2_focalloss_v2"' in code
    assert "RUN_FINAL_TEST = False" in code
    assert "len(experiments) == 4" in code
    assert "prepare_stage2_colab" in code
    assert "run_stage2_colab" in code
    assert "prepared_context=STAGE_CONTEXT" in code
    assert "stage2_colab_v2" in code
    assert "stage2_focalloss_v2_results.zip" in code
    assert "consensus_test" not in code


def test_stage2_local_densenet_notebook_contract():
    notebook_name = "04_1_local_densenet121_focalloss.ipynb"
    notebook = json.loads(
        (Path("notebooks") / notebook_name).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    compile(code, notebook_name, "exec")

    assert 'LOCAL_STAGE_ID = "stage2_focalloss_v2_local_fp32"' in code
    assert 'MODEL_NAME = "densenet121"' in code
    assert 'WEIGHTING = "sqrtcap3"' in code
    assert "RUN_FINAL_TEST = False" in code
    assert "len(experiments) == 4" in code
    assert 'REPO_DIR / "checkpoint" / LOCAL_STAGE_ID / MODEL_SLUG' in code
    assert (
        'REPO_DIR / "data" / "processed" / "consensus_v1_224"'
        in code
    )
    assert (
        'REPO_DIR / "data" / "protocols" / '
        '"vindr_s8515_exttest_v1"'
        in code
    )
    assert "TRAIN_BATCH_SIZE = 8" in code
    assert "VALIDATION_BATCH_SIZE = 16" in code
    assert "NUM_WORKERS = 0" in code
    assert '"use_amp": False' in code
    assert '"amp": False' in code
    assert '"precision": "float32"' in code
    assert 'removesuffix("_amp") + "_fp32"' in code
    assert "_run_completion(experiment, config, CHECKPOINT_ROOT, amp_enabled=False)" in code
    assert 'environment["PYTHONIOENCODING"] = "utf-8"' in code
    assert 'environment["PYTHONUTF8"] = "1"' in code
    assert 'encoding="utf-8", errors="replace"' in code
    assert "_run_subprocess_streaming" in code
    assert 'local_subprocess_root = LOCAL_STAGE_ROOT / "subprocess_logs"' in code
    assert 'read_text(encoding="utf-8", errors="replace")' in code
    assert '"state": "interrupted"' in code
    assert "Path.cwd().resolve()" in code
    assert "DATH_REPO_ROOT" in code
    assert "google.colab" not in code
    assert "drive.mount" not in code
    assert 'Path("/content")' not in code
    assert "consensus_test" not in code


@pytest.mark.parametrize(
    "notebook_name,stage_id,model_name,weighting,positive_weight,weight_enum,export_name",
    [
        (
            "04_2_local_resnet50_focalloss.ipynb",
            "stage2_focalloss_v2_local_fp32_resnet50",
            "resnet50",
            "raw",
            '{"mode": "raw_ratio", "max_value": None}',
            "ResNet50_Weights.DEFAULT",
            "04_2_local_resnet50_stage2_focalloss_v2_fp32_results.zip",
        ),
        (
            "04_3_local_vit_b16_focalloss.ipynb",
            "stage2_focalloss_v2_local_fp32_vit_b16",
            "vit_b_16",
            "sqrtcap3",
            '{"mode": "sqrt_ratio_capped", "max_value": 3.0}',
            "ViT_B_16_Weights.DEFAULT",
            "04_3_local_vit_b16_stage2_focalloss_v2_fp32_results.zip",
        ),
    ],
)
def test_stage2_local_fp32_notebook_contract(
    notebook_name,
    stage_id,
    model_name,
    weighting,
    positive_weight,
    weight_enum,
    export_name,
):
    notebook = json.loads(
        (Path("notebooks") / notebook_name).read_text(encoding="utf-8")
    )
    code_cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    compile(code, notebook_name, "exec")

    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)
    assert f'LOCAL_STAGE_ID = "{stage_id}"' in code
    assert f'MODEL_NAME = "{model_name}"' in code
    assert f'MODEL_SLUG = "{model_name}"' in code
    assert f'WEIGHTING = "{weighting}"' in code
    assert f"POSITIVE_WEIGHT = {positive_weight}" in code
    assert weight_enum in code
    assert export_name in code
    assert "RUN_FINAL_TEST = False" in code
    assert "TRAIN_BATCH_SIZE = 8" in code
    assert "VALIDATION_BATCH_SIZE = 16" in code
    assert "NUM_WORKERS = 0" in code
    assert '"use_amp": False' in code
    assert '"amp": False' in code
    assert '"precision": "float32"' in code
    assert 'removesuffix("_amp") + "_fp32"' in code
    assert code.count("amp_enabled=False") == 2
    assert 'environment["PYTHONIOENCODING"] = "utf-8"' in code
    assert 'environment["PYTHONUTF8"] = "1"' in code
    assert 'REPO_DIR / "checkpoint" / LOCAL_STAGE_ID / MODEL_SLUG' in code
    assert 'LOCAL_STAGE_ROOT = REPO_DIR / "outputs" / LOCAL_STAGE_ID' in code
    assert "len(experiments) == 4" in code
    assert '"state": "interrupted"' in code
    assert "google.colab" not in code
    assert "drive.mount" not in code
    assert 'Path("/content")' not in code
    assert "consensus_test" not in code


def test_stage2_local_fp32_stage_roots_are_disjoint():
    notebook_names = [
        "04_1_local_densenet121_focalloss.ipynb",
        "04_2_local_resnet50_focalloss.ipynb",
        "04_3_local_vit_b16_focalloss.ipynb",
    ]
    expected_stage_ids = {
        "stage2_focalloss_v2_local_fp32",
        "stage2_focalloss_v2_local_fp32_resnet50",
        "stage2_focalloss_v2_local_fp32_vit_b16",
    }
    observed = set()
    for notebook_name in notebook_names:
        notebook = json.loads(
            (Path("notebooks") / notebook_name).read_text(encoding="utf-8")
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        stage_line = next(
            line for line in code.splitlines()
            if line.startswith("LOCAL_STAGE_ID = ")
        )
        observed.add(stage_line.split('"')[1])
    assert observed == expected_stage_ids


def test_stage2_model_subtrees_and_run_names_are_disjoint():
    specs = [
        ("densenet121", "densenet121", "sqrtcap3", {"mode": "sqrt_ratio_capped", "max_value": 3.0}),
        ("resnet50", "resnet50", "raw", {"mode": "raw_ratio", "max_value": None}),
        ("vit_b_16", "vit_b_16", "sqrtcap3", {"mode": "sqrt_ratio_capped", "max_value": 3.0}),
    ]
    run_names = set()
    model_slugs = set()
    for model, slug, weighting, positive_weight in specs:
        experiments = build_stage2_experiments(model, slug, "config.yaml", weighting, positive_weight)
        assert len(experiments) == 4
        model_slugs.add(slug)
        for experiment in experiments:
            assert experiment["run_name"] not in run_names
            run_names.add(experiment["run_name"])
    assert len(model_slugs) == 3 and len(run_names) == 12
