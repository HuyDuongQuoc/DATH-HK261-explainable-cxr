import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.train import (
    build_training_protocol,
    validate_resume_training_protocol,
)
from src.analysis.stage1 import analyze_stage1
from src.losses.classification_loss import (
    LEGACY_RAW_LOSS_PROTOCOL,
    calculate_pos_weight,
    validate_checkpoint_loss_protocol,
)
from src.training.engine import run_epoch
from src.utils.config import validate_loss_config


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"mode": "raw_ratio", "max_value": None}, [3.0, 1.0]),
        ({"mode": "sqrt_ratio", "max_value": None}, [np.sqrt(3.0), 1.0]),
        ({"mode": "sqrt_ratio_capped", "max_value": 1.5}, [1.5, 1.0]),
    ],
)
def test_positive_weight_modes(config, expected):
    labels = np.asarray([[1, 1], [0, 0], [0, 1], [0, 0]], dtype=float)
    weight, metadata = calculate_pos_weight(labels, "cpu", config)
    assert torch.allclose(weight, torch.tensor(expected, dtype=torch.float32))
    assert metadata["positive_count"] == [1.0, 2.0]
    assert metadata["negative_count"] == [3.0, 2.0]


def test_loss_config_validation_and_legacy_protocol():
    training = {}
    validate_loss_config(training)
    assert training["loss"] == LEGACY_RAW_LOSS_PROTOCOL

    with pytest.raises(ValueError, match="max_value"):
        validate_loss_config({
            "loss": {
                "name": "weighted_bce",
                "positive_weight": {"mode": "sqrt_ratio_capped", "max_value": 0},
            }
        })
    with pytest.raises(ValueError, match="must be null"):
        validate_loss_config({
            "loss": {
                "name": "weighted_bce",
                "positive_weight": {"mode": "raw_ratio", "max_value": 5},
            }
        })


def test_training_epoch_can_skip_prediction_collection():
    samples = [
        {"image": torch.ones(3, 2, 2), "labels": torch.tensor([1.0]), "image_id": "a"},
        {"image": torch.zeros(3, 2, 2), "labels": torch.tensor([0.0]), "image_id": "b"},
    ]
    loader = DataLoader(samples, batch_size=2)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = run_epoch(
        model,
        loader,
        torch.nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        optimizer=optimizer,
        collect_predictions=False,
    )
    assert "targets" not in result
    assert "probabilities" not in result
    assert result["peak_gpu_memory_allocated_gb"] == 0
    assert result["peak_gpu_memory_reserved_gb"] == 0


def test_resume_requires_matching_training_and_loss_protocol():
    training = {
        "batch_size": 8,
        "validation_batch_size": 64,
        "num_workers": 2,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "monitor": "macro_average_precision",
    }
    augmentation = {"enabled": True, "config": {"rotation_degrees": 7.0}}
    protocol = build_training_protocol(
        training,
        augmentation,
        LEGACY_RAW_LOSS_PROTOCOL,
        amp_enabled=True,
    )
    checkpoint = {"training_protocol": protocol, "loss_protocol": LEGACY_RAW_LOSS_PROTOCOL}
    validate_resume_training_protocol(checkpoint, protocol)
    validate_checkpoint_loss_protocol(checkpoint, LEGACY_RAW_LOSS_PROTOCOL)
    with pytest.raises(ValueError, match="training protocol"):
        validate_resume_training_protocol(
            checkpoint,
            {**protocol, "batch_size": 16},
        )


def _write_run(root, checkpoint_root, experiment, score):
    findings = ["a", "b"]
    result_dir = root / experiment["run_name"]
    result_dir.mkdir(parents=True)
    checkpoint_dir = checkpoint_root / experiment["run_name"]
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pt").write_bytes(b"checkpoint")
    per_class = [
        {"finding": finding, "threshold": 0.5, "auroc": score + 0.2, "average_precision": score,
         "precision": score, "recall": 0.5, "f1": score, "positive_count": 2, "negative_count": 2}
        for finding in findings
    ]
    metrics = {
        "per_class": per_class,
        "macro_auroc": score + 0.2,
        "macro_average_precision": score,
        "macro_precision": score,
        "macro_recall": 0.5,
        "macro_f1": score,
        "checkpoint_epoch": 1,
    }
    (result_dir / "best_val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    predictions = pd.DataFrame({
        "image_id": ["i1", "i2", "i3", "i4"],
        "target::a": [1, 0, 1, 0], "probability::a": [0.9, 0.2, 0.8, 0.1],
        "target::b": [0, 1, 0, 1], "probability::b": [0.1, 0.9, 0.2, 0.8],
    })
    predictions.to_csv(result_dir / "best_val_predictions.csv", index=False)
    loss = {
        "positive_count": [2, 2], "negative_count": [2, 2],
        "raw_pos_weight": [1, 1], "effective_pos_weight": [1, 1],
    }
    (result_dir / "loss_metadata.json").write_text(json.dumps(loss), encoding="utf-8")
    pd.DataFrame([{
        "epoch": 1, "val_macro_ap": score, "train_seconds": 1, "val_seconds": 1,
        "train_images_per_second": 2, "peak_gpu_memory_gb": 0,
        "peak_gpu_memory_allocated_gb": 0, "peak_gpu_memory_reserved_gb": 0,
    }]).to_csv(result_dir / "history.csv", index=False)


def test_stage1_analysis_selects_complete_model_and_writes_reports(tmp_path):
    experiments = []
    scores = {"raw": 0.50, "sqrt": 0.52, "sqrtcap5": 0.55, "sqrtcap3": 0.51}
    checkpoint_root = tmp_path / "checkpoints"
    run_root = tmp_path / "runs"
    for weighting, score in scores.items():
        experiment = {"run_name": f"resnet_{weighting}", "model_name": "resnet50", "weighting": weighting}
        experiments.append(experiment)
        _write_run(run_root, checkpoint_root, experiment, score)
    result = analyze_stage1(experiments, run_root, checkpoint_root)
    assert result["selection"]["resnet50"]["winner"] == "sqrtcap5"
    assert (run_root / "analysis" / "stage1_analysis_report.md").is_file()
    assert (run_root / "analysis" / "figures" / "01_macro_auprc_by_weighting.png").is_file()


def test_notebook03_contract():
    notebook = json.loads(Path("notebooks/03_testing.ipynb").read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'NOTEBOOK_GROUP = "03_testing"' in source
    assert 'STAGE_ID = "stage1_pos_weight_v1"' in source
    assert "assert len(EXPERIMENTS) == 12" in source
    assert 'RUN_FINAL_TEST = False' in source
    assert '"validation_batch_size": 64' in source
    assert 'MAX_NEW_RUNS_PER_SESSION = None' in source
    assert "analyze_stage1" in source
    assert "gpu_telemetry_" in source
    assert "External test is intentionally disabled in Stage 1" in source
    assert "03_testing_stage1_pos_weight_v1_results.zip" in source
