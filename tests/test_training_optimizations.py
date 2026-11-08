import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import CXRDataset
from src.data.loader import build_loader_kwargs
from src.training.engine import run_epoch
from src.utils.checkpoint import load_checkpoint, save_checkpoint


class TinyDataset(Dataset):
    def __init__(self, size=8):
        generator = torch.Generator().manual_seed(42)
        self.images = torch.randn(size, 3, 4, 4, generator=generator)
        self.labels = torch.randint(0, 2, (size, 2), generator=generator).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "labels": self.labels[index],
            "image_id": f"image_{index}",
        }


def test_classification_dataset_skips_npz_loading(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("L", (8, 8), color=128).save(image_dir / "sample.png")
    manifest = pd.DataFrame(
        [
            {
                "image_id": "sample",
                "image_path": "images/sample.png",
                "mask_path": "masks/does-not-exist.npz",
                "finding": 1,
            }
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("np.load must not be called")
        ),
    )
    dataset = CXRDataset(
        manifest_path=manifest_path,
        processed_root=tmp_path,
        findings=["finding"],
        image_size=8,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        load_masks=False,
    )
    sample = dataset[0]

    assert set(sample) == {"image", "labels", "image_id"}
    assert sample["image"].shape == (3, 8, 8)
    assert sample["labels"].tolist() == [1.0]


def test_loader_kwargs_only_use_worker_options_when_workers_are_positive():
    base = {
        "batch_size": 8,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    zero_workers = build_loader_kwargs(
        {**base, "num_workers": 0},
        pin_memory=False,
    )
    assert "persistent_workers" not in zero_workers
    assert "prefetch_factor" not in zero_workers

    two_workers = build_loader_kwargs(
        {**base, "num_workers": 2},
        pin_memory=True,
    )
    assert two_workers["persistent_workers"] is True
    assert two_workers["prefetch_factor"] == 2
    assert two_workers["pin_memory"] is True


def test_fp32_cpu_epoch_reports_timing_and_validation_does_not_update():
    device = torch.device("cpu")
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_result = run_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=optimizer,
        use_amp=True,
        show_progress=False,
    )
    before_validation = [parameter.detach().clone() for parameter in model.parameters()]
    val_result = run_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=None,
        use_amp=True,
        show_progress=False,
    )

    assert np.isfinite(train_result["loss"])
    assert train_result["elapsed_seconds"] > 0
    assert train_result["images_per_second"] > 0
    assert val_result["peak_gpu_memory_gb"] == 0
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_validation, model.parameters())
    )


def test_checkpoint_round_trip_includes_scaler_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "decision_thresholds": {"finding": 0.42},
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, path)
    loaded = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device="cpu",
    )
    assert "amp_scaler_state_dict" in loaded
    assert loaded["decision_thresholds"] == {"finding": 0.42}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_epoch_is_finite():
    device = torch.device("cuda")
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2)).to(device)
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False, pin_memory=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    result = run_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=optimizer,
        scaler=scaler,
        use_amp=True,
        show_progress=False,
    )
    assert np.isfinite(result["loss"])
    assert result["peak_gpu_memory_gb"] > 0


def test_notebook_declares_s8515_threshold_and_external_test_contract():
    notebook = json.loads(
        open("notebooks/02_experiment.ipynb", encoding="utf-8").read()
    )
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "requirements-common.txt" in source
    assert "AMP FP16 probe: OK" in source
    assert 'EXPERIMENT_GROUP = "aug_auprc_v1"' in source
    assert "aug_auprc_s8515_exttest_v1_seed42_amp" in source
    assert 'MONITOR_NAME = "macro_average_precision"' in source
    assert "configs/resnet50_aug_auprc.yaml" in source
    assert "configs/densenet121_aug_auprc.yaml" in source
    assert "configs/vit_aug_auprc.yaml" in source
    assert 'REPO_BRANCH = "update_data"' in source
    assert 'TRAIN_ARCHIVE = DRIVE_ROOT / "datasets" / "consensus_v1_224.zip"' in source
    assert 'TEST_ARCHIVE = DRIVE_ROOT / "datasets" / "consensus_test_v1_224.zip"' in source
    assert "EXPECTED_TRAIN_ROWS = 12_750" in source
    assert "EXPECTED_VAL_ROWS = 2_250" in source
    assert "EXPECTED_TEST_IMAGES = 3_000" in source
    assert "RUN_FINAL_TEST = False" in source
    assert 'CHECKPOINT_ROOT = DRIVE_ROOT / "checkpoints" / EXPERIMENT_GROUP' in source
    assert 'RUN_LOG_ROOT = DRIVE_ROOT / "run_logs" / EXPERIMENT_GROUP' in source
    assert "02_experiment_aug_auprc_v1_results.zip" in source
    assert '"validation_macro_average_precision"' in source
    assert '"selection_metric": MONITOR_NAME' in source
    assert 'config["data"]["external_test"]["processed_root"]' in source
    assert 'config["data"]["external_test"]["manifest_path"]' in source
    assert '"threshold_source"] != "checkpoint"' in source
    assert '"--split", "test"' in source
    assert "final_test_plan.json" in source
    assert "external_test_figures" in source
    assert "def metadata_text_sha256(path):" in source
    assert 'data.replace(b"\\n", b"\\r\\n")' in source
    assert 'metadata_text_sha256(PROTOCOL_ROOT / "external_test_manifest.csv")' in source
    assert 'metadata_text_sha256(REPO_DIR / "data" / "raw" / "test.csv")' in source
    assert "EXPECTED_TEST_IMAGES = 2_250" not in source
    assert "TEST_THRESHOLD = 0.5" not in source
    assert "Internal-test" not in source
    assert "internal_test" not in source
    assert "raise SystemExit" not in source
    assert "enumerate(EXPERIMENTS, start=1)" in source
    assert "evaluate_best_checkpoint" not in source
