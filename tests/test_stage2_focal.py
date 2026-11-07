import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from scripts.train import build_training_protocol, validate_resume_training_protocol
from src.analysis.stage2 import analyze_stage2
from src.experiments.stage2_colab import (
    STAGE1_RUNTIME_CONTRACT,
    build_stage2_experiments,
    create_or_validate_contract,
    validate_export_members,
)
from src.losses.classification_loss import (
    SigmoidFocalLoss,
    build_classification_loss,
    build_loss_from_checkpoint,
    validate_checkpoint_loss_protocol,
)
from src.utils.config import validate_loss_config


def _focal_protocol(gamma=1.0, mode="sqrt_ratio_capped", maximum=3.0):
    return {
        "name": "sigmoid_focal",
        "gamma": gamma,
        "positive_weight": {"mode": mode, "max_value": maximum},
    }


def test_focal_gamma_zero_matches_weighted_bce():
    logits = torch.tensor([[1.2, -0.7], [-0.1, 2.1]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pos_weight = torch.tensor([3.0, 1.5])
    focal = SigmoidFocalLoss(pos_weight, gamma=0.0)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    assert torch.allclose(focal, bce, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5])
def test_focal_forward_backward_is_finite(gamma):
    logits = torch.randn(8, 14, requires_grad=True)
    targets = torch.randint(0, 2, (8, 14), dtype=torch.float32)
    loss = SigmoidFocalLoss(torch.linspace(1, 5, 14), gamma)(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is unavailable")
@pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5])
def test_focal_amp_is_finite(gamma):
    logits = torch.randn(8, 14, device="cuda", requires_grad=True)
    targets = torch.randint(0, 2, (8, 14), device="cuda", dtype=torch.float32)
    criterion = SigmoidFocalLoss(torch.ones(14, device="cuda"), gamma).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("gamma", [None, -1.0, 0.0, float("nan"), 5.1])
def test_focal_config_rejects_invalid_gamma(gamma):
    training = {
        "loss": {
            "name": "sigmoid_focal",
            "gamma": gamma,
            "positive_weight": {"mode": "raw_ratio", "max_value": None},
        }
    }
    with pytest.raises(ValueError, match="gamma"):
        validate_loss_config(training)


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


@pytest.mark.parametrize(
    "notebook_name,model_name,weighting",
    [
        ("04_1_densenet121_focalloss.ipynb", "densenet121", "sqrtcap3"),
        ("04_2_resnet50_focalloss.ipynb", "resnet50", "raw"),
        ("04_3_vit_b16_focalloss.ipynb", "vit_b_16", "sqrtcap3"),
    ],
)
def test_stage2_notebook_contract(notebook_name, model_name, weighting):
    notebook = json.loads((Path("notebooks") / notebook_name).read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    compile(code, notebook_name, "exec")
    assert f'"model_name": "{model_name}"' in code
    assert f'"weighting": "{weighting}"' in code
    assert "RUN_FINAL_TEST = False" in code
    assert "len(experiments) == 4" in code
    assert "run_stage2_colab" in code
    assert "consensus_test" not in code


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
