import numpy as np
import torch

from src.metrics.localization import iou_column
from src.models.factory import build_model
from src.xai.evaluation import collect_localization_records, summarize_by_finding

FINDINGS = [f"f{i}" for i in range(14)]


def fake_batches(valid_pairs, batch=2):
    torch.manual_seed(0)
    union = torch.zeros(batch, 14, 224, 224)
    union[:, :, :112, :112] = 1
    valid = torch.zeros(batch, 14, dtype=torch.bool)
    for sample, cls in valid_pairs:
        valid[sample, cls] = True
    return [{
        "image": torch.randn(batch, 3, 224, 224),
        "union_masks": union,
        "mask_valid": valid,
        "image_id": [f"img{i}" for i in range(batch)],
    }]


def test_records_one_row_per_valid_pair_with_expected_columns():
    model = build_model("resnet50", 14, pretrained=False)
    batches = fake_batches([(0, 1), (0, 4), (1, 4)])
    records = collect_localization_records(
        model, "resnet50", batches, FINDINGS, torch.device("cpu"),
        thresholds=[0.3, 0.5],
    )
    assert len(records) == 3
    assert sorted(records["finding"]) == ["f1", "f4", "f4"]
    assert {"pointing_hit", "eib", iou_column(0.3), iou_column(0.5)} <= set(records.columns)
    assert records["eib"].between(0, 1).all()
    assert records[iou_column(0.3)].between(0, 1).all()
    assert set(records["pointing_hit"]) <= {0, 1}


def test_records_empty_when_no_valid_pairs():
    model = build_model("resnet50", 14, pretrained=False)
    records = collect_localization_records(
        model, "resnet50", fake_batches([]), FINDINGS, torch.device("cpu"),
        thresholds=[0.5],
    )
    assert records.empty


def test_summary_by_finding_has_macro_row():
    model = build_model("resnet50", 14, pretrained=False)
    records = collect_localization_records(
        model, "resnet50", fake_batches([(0, 1), (1, 4)]), FINDINGS,
        torch.device("cpu"), thresholds=[0.5],
    )
    table = summarize_by_finding(records, FINDINGS, 0.5)
    assert len(table) == 15
    assert table.iloc[-1]["finding"] == "MACRO"
    assert table.iloc[-1]["n_pairs"] == 2
    assert np.isnan(table.loc[table["finding"] == "f0", "iou"].iloc[0])
