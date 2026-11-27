import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import rq1, rq2
from src.analysis.rq_data import collect_runs, long_table, markdown_table

FINDINGS = ["A", "B", "C", "D"]
ROOT = Path(__file__).resolve().parents[1]


def make_run(root, name, model, lam, seed, auroc, eib, n_images=40):
    """Synthetic run with the given per-finding AUROC and EIB."""
    rng = np.random.default_rng(seed + int(lam * 10))
    run = root / name
    (run / "res").mkdir(parents=True)
    (run / "res" / "config.json").write_text(json.dumps({
        "seed": seed,
        "model": {"name": model},
        "localization_loss": {"enabled": lam > 0, "lambda_loc": lam or 0.25},
        "training": {"loss": {"name": "weighted_bce"}},
    }))
    (run / "cls_test").mkdir()
    (run / "cls_test" / "test_metrics.json").write_text(json.dumps({
        "per_class": [
            {"finding": f, "auroc": a, "average_precision": a / 2}
            for f, a in zip(FINDINGS, auroc)
        ]
    }))
    ids = [f"img{i}" for i in range(n_images)]
    targets = np.random.default_rng(0).integers(0, 2, (n_images, len(FINDINGS)))  # same labels for every run
    targets[0], targets[1] = 0, 1
    preds = pd.DataFrame({"image_id": ids})
    for j, f in enumerate(FINDINGS):
        preds[f"target::{f}"] = targets[:, j]
        preds[f"probability::{f}"] = np.clip(targets[:, j] * auroc[j] + rng.normal(0, 0.3, n_images), 0, 1)
    preds.to_csv(run / "cls_test" / "test_predictions.csv", index=False)

    for folder, scale in (("loc_test_gradcam", 1.0), ("loc_test_occlusion", 0.8)):
        (run / folder).mkdir()
        records = []
        for f, e in zip(FINDINGS, eib):
            for i in ids[:10]:
                value = float(np.clip(e * scale + rng.normal(0, 0.02), 0, 1))
                records.append({"image_id": i, "finding": f, "eib": value,
                                "pointing_hit": int(value > 0.3), "eib_minus_prior": value - 0.1})
        rec = pd.DataFrame(records)
        rec.to_csv(run / folder / "test_localization_records.csv", index=False)
        by = rec.groupby("finding").agg(n_pairs=("eib", "size"), pointing_hit_rate=("pointing_hit", "mean"),
                                        eib=("eib", "mean"), eib_minus_prior=("eib_minus_prior", "mean")).reset_index()
        by.to_csv(run / folder / "test_localization_by_finding.csv", index=False)
    return run


@pytest.fixture()
def runs_root(tmp_path):
    base_auc = [0.9, 0.8, 0.7, 0.6]
    # ResNet: high AUROC, low EIB; ViT: low AUROC, high EIB -> negative within-finding correlation
    for seed in (42, 43):
        make_run(tmp_path, f"r50_base_s{seed}", "resnet50", 0, seed, [a + 0.05 for a in base_auc], [0.1, 0.1, 0.1, 0.1])
        make_run(tmp_path, f"d121_base_s{seed}", "densenet121", 0, seed, base_auc, [0.2, 0.2, 0.2, 0.2])
        make_run(tmp_path, f"vitgap_base_s{seed}", "vit_b_16_gap", 0, seed, [a - 0.05 for a in base_auc], [0.3, 0.3, 0.3, 0.3])
        make_run(tmp_path, f"r50_l1_s{seed}", "resnet50", 1.0, seed, [a + 0.04 for a in base_auc], [0.4, 0.4, 0.4, 0.4])
    return tmp_path


def test_collect_and_long_table(runs_root):
    runs = collect_runs(runs_root)
    assert len(runs) == 8
    assert set(runs.loc[runs["run"] == "r50_l1_s42", ["lambda", "seed", "arch"]].iloc[0]) == {1.0, 42, "ResNet-50"}
    table, missing = long_table(runs, "test", "gradcam")
    assert not missing and len(table) == 8 * len(FINDINGS)


def test_rq1_detects_negative_within_finding_correlation(runs_root):
    runs = collect_runs(runs_root)
    table, _ = long_table(runs, "test", "gradcam")
    arch = rq1.architecture_table(table)
    assert set(arch.index) == {"ResNet-50", "DenseNet-121", "ViT-B/16 (GAP)"}
    assert arch.loc["ResNet-50", "auroc_rank"] == 1 and arch.loc["ResNet-50", "eib_rank"] == 3
    corr = rq1.correlation_table(table, n_boot=200)
    row = corr[(corr.cls_metric == "auroc") & (corr.loc_metric == "eib")].iloc[0]
    assert row.within_finding_spearman < -0.5
    assert row.runs_per_finding == 6
    inter = rq1.intervention_table(table)
    assert len(inter) == 2
    assert np.allclose(inter["delta_auroc"], -0.01, atol=1e-9)
    assert (inter["delta_eib"] > 0.25).all()
    md = rq1.to_markdown(arch, corr, inter, "test", "gradcam")
    assert "within_finding_spearman" in md and "| arch |" in md


def test_rq2_pairs_and_paired_table(runs_root):
    runs = collect_runs(runs_root)
    pairs = rq2.pairs(runs)
    assert sorted((b["run"], g["run"]) for b, g in pairs) == [
        ("r50_base_s42", "r50_l1_s42"), ("r50_base_s43", "r50_l1_s43")
    ]
    paired = rq2.paired_table(runs, FINDINGS, n_boot=100)
    eib = paired[(paired.method == "occlusion") & (paired.metric == "eib")]
    assert len(eib) == 2 and (eib["diff"] > 0.2).all() and (eib["ci_low"] > 0).all()
    assert set(paired["method"]) == {"classification", "gradcam", "occlusion"}
    per_finding = rq2.per_finding_table(runs)
    assert set(per_finding["finding"]) == set(FINDINGS)
    assert "delta_eib" in per_finding and "delta_auroc" in per_finding
    assert "Paired comparison" in rq2.to_markdown(paired, per_finding)


def test_markdown_table_handles_nan():
    text = markdown_table(pd.DataFrame({"a": [1.23456, np.nan]}), index=False)
    assert text.splitlines()[0] == "| a |" and "| 1.2346 |" in text


def test_cli_end_to_end(runs_root, tmp_path):
    out = tmp_path / "rq_out"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rq_analysis.py"), str(runs_root),
         "--output-dir", str(out), "--n-boot", "50"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    for name in ("runs.csv", "rq1_gradcam.md", "rq1_occlusion_correlation.csv", "rq2.md", "rq2_paired.csv"):
        assert (out / name).is_file(), name
    rq1_archs = set(pd.read_csv(out / "rq1_gradcam_long.csv")["arch"])
    assert rq1_archs == {"ResNet-50", "DenseNet-121"}


def test_normalized_drops_remove_logit_scale():
    rng = np.random.default_rng(0)
    logit = rng.normal(size=50)
    base = pd.DataFrame({
        "image_id": [f"i{k}" for k in range(50)], "finding": "A",
        "logit": logit, "drop_inside": 0.5 * logit + 1.0, "drop_outside": 0.1 * logit,
    })
    scaled = base.assign(**{c: base[c] * 4 for c in ("logit", "drop_inside", "drop_outside")})
    a, b = rq2.add_normalized_drops(base), rq2.add_normalized_drops(scaled)
    assert np.allclose(a["drop_inside_norm"], b["drop_inside_norm"])
    assert np.allclose(a["drop_outside_norm"], b["drop_outside_norm"])
    assert rq2.add_normalized_drops(base.drop(columns="logit")).columns.tolist() == [
        "image_id", "finding", "drop_inside", "drop_outside"
    ]
