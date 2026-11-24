"""
So sánh paired hai run (A = tham chiếu, thường là baseline; B = run cần đánh giá)
trên cùng split, với bootstrap theo ảnh.

Ví dụ:
    python scripts/compare_runs.py \
        --config configs/resnet50_location.yaml \
        --a-loc outputs/full/runs/r50_base/loc_test/test_localization_records.csv \
        --b-loc outputs/full/runs/r50_l1/loc_test/test_localization_records.csv \
        --a-pred outputs/full/runs/r50_base/cls_test/test_predictions.csv \
        --b-pred outputs/full/runs/r50_l1/cls_test/test_predictions.csv \
        --output outputs/comparisons/r50_base_vs_l1_test.json

Lưu ý: chỉ so các run được đánh giá bằng cùng --method và cùng ngưỡng IoU
(dùng iou_mean hoặc --iou-threshold cố định khi chạy evaluate_localization.py).
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.localization_stats import (
    load_predictions,
    paired_auroc_diff,
    paired_localization_diff,
)
from src.utils.config import load_config

DEFAULT_LOC_METRICS = (
    "pointing_hit",
    "eib",
    "eib_over_area",
    "eib_minus_prior",
    "iou_mean",
    "drop_inside",
    "drop_outside",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="Để lấy danh sách finding")
    parser.add_argument("--a-loc")
    parser.add_argument("--b-loc")
    parser.add_argument("--a-pred")
    parser.add_argument("--b-pred")
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_LOC_METRICS))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not ((args.a_loc and args.b_loc) or (args.a_pred and args.b_pred)):
        parser.error("Cần ít nhất một cặp --a-loc/--b-loc hoặc --a-pred/--b-pred")

    findings = load_config(args.config)["data"]["target_findings"]
    result = {"a": {}, "b": {}, "localization": [], "classification": []}

    if args.a_loc and args.b_loc:
        result["a"]["localization_records"] = args.a_loc
        result["b"]["localization_records"] = args.b_loc
        records_a = pd.read_csv(args.a_loc)
        records_b = pd.read_csv(args.b_loc)
        for metric in args.metrics:
            if metric in records_a.columns and metric in records_b.columns:
                result["localization"].append(
                    paired_localization_diff(
                        records_a, records_b, metric,
                        n_boot=args.n_boot, seed=args.seed,
                    )
                )

    if args.a_pred and args.b_pred:
        result["a"]["predictions"] = args.a_pred
        result["b"]["predictions"] = args.b_pred
        result["classification"].append(
            paired_auroc_diff(
                load_predictions(args.a_pred, findings),
                load_predictions(args.b_pred, findings),
                n_boot=args.n_boot,
                seed=args.seed,
            )
        )

    for row in result["localization"] + result["classification"]:
        print(
            f"{row['metric']:>16}: A={row['a']:.4f} B={row['b']:.4f} "
            f"diff={row['diff']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] "
            f"p={row['p_two_sided']:.3f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
