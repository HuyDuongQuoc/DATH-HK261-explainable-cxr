"""
Phân tích RQ1 và RQ2 từ các run đã tải về (vd từ volume Modal).

    ~/.venvs/modal/bin/python -m modal volume get cxr-runs / outputs/modal_runs/volume
    python scripts/rq_analysis.py outputs/modal_runs/volume --output-dir outputs/rq

Chỉ dùng các run có res/config.json; model, λ, seed đọc từ config. Mặc định bỏ
các run ablation loss (tên bắt đầu bằng "loss_") để mọi run cùng một classification loss.

Đầu ra (trong --output-dir):
    runs.csv                         danh sách run được dùng
    rq1_<method>_long.csv            bảng run × finding
    rq1_<method>_{architecture,correlation,intervention}.csv, rq1_<method>.md
    rq2_paired.csv, rq2_per_finding.csv, rq2.md
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import rq1, rq2
from src.analysis.rq_data import collect_runs, long_table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", help="Thư mục chứa các run")
    parser.add_argument("--output-dir", default="outputs/rq")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--include-loss-ablation", action="store_true")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    runs = collect_runs(args.root)
    if runs.empty:
        raise SystemExit(f"Không tìm thấy run có res/config.json trong {args.root}")
    if not args.include_loss_ablation:
        runs = runs[~runs["run"].str.startswith("loss_")]
    if runs["loss"].nunique() > 1:
        print("CẢNH BÁO: các run dùng classification loss khác nhau:")
        print(runs.groupby("loss")["run"].apply(list).to_string())

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs.drop(columns=["loss"]).to_csv(output / "runs.csv", index=False)
    print(runs[["run", "arch", "lambda", "seed"]].to_string(index=False))

    methods = ["gradcam", "occlusion"] if args.split == "test" else ["gradcam"]
    for method in methods:
        table, missing = long_table(runs, args.split, method)
        if missing:
            print(f"[{method}] thiếu kết quả, bỏ qua: {missing}")
        if table.empty or table.loc[table["lambda"] == 0, "arch"].nunique() < 2:
            print(f"[{method}] chưa đủ baseline của ≥2 kiến trúc cho RQ1")
            continue
        arch = rq1.architecture_table(table)
        corr = rq1.correlation_table(table, n_boot=args.n_boot)
        intervention = rq1.intervention_table(table)
        table.to_csv(output / f"rq1_{method}_long.csv", index=False)
        arch.to_csv(output / f"rq1_{method}_architecture.csv")
        corr.to_csv(output / f"rq1_{method}_correlation.csv", index=False)
        intervention.to_csv(output / f"rq1_{method}_intervention.csv", index=False)
        report = rq1.to_markdown(arch, corr, intervention, args.split, method)
        (output / f"rq1_{method}.md").write_text(report, encoding="utf-8")
        print(report)

    if args.split == "test":
        paired = rq2.paired_table(runs, n_boot=args.n_boot)  # finding lấy từ file predictions
        per_finding = rq2.per_finding_table(runs)
        paired.to_csv(output / "rq2_paired.csv", index=False)
        per_finding.to_csv(output / "rq2_per_finding.csv", index=False)
        report = rq2.to_markdown(paired, per_finding)
        (output / "rq2.md").write_text(report, encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
