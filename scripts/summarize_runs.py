"""
Tổng hợp kết quả validation (và test nếu có) của các run, gom theo cấu hình
(tên run bỏ hậu tố _s<seed>), trả về mean ± sd qua các seed.

    python scripts/summarize_runs.py outputs/modal_runs --pattern 'loss_r50_*'
    python scripts/summarize_runs.py outputs/modal_runs --output outputs/comparisons/loss_ablation.csv

Đọc, nếu có, trong mỗi <run>/:
    res/best_val_metrics.json          macro_auroc, macro_average_precision, macro_f1
    loc/localization_summary.json      val: pointing, eib (+ eib_minus_prior nếu có)
    cls_test/test_metrics.json         test macro_auroc, macro_average_precision
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SEED_SUFFIX = re.compile(r"_s(\d+)$")


def _read(*paths):
    """Đọc file JSON đầu tiên tồn tại (hỗ trợ cả bố cục res/, loc/ lẫn bố cục phẳng khi tải về)."""
    for path in paths:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def collect(root, pattern):
    rows = []
    for run_dir in sorted(Path(root).glob(pattern)):
        if not run_dir.is_dir():
            continue
        match = SEED_SUFFIX.search(run_dir.name)
        row = {
            "run": run_dir.name,
            "config": SEED_SUFFIX.sub("", run_dir.name),
            "seed": int(match.group(1)) if match else None,
        }
        val = _read(run_dir / "res" / "best_val_metrics.json", run_dir / "best_val_metrics.json")
        if val is None:
            row["status"] = "incomplete"
            rows.append(row)
            continue
        row["status"] = "ok"
        row["best_epoch"] = val.get("checkpoint_epoch")
        for key in ("macro_auroc", "macro_average_precision", "macro_f1"):
            row[f"val_{key}"] = val.get(key)

        loc = _read(run_dir / "loc" / "localization_summary.json", run_dir / "localization_summary.json")
        if loc and "val" in loc:
            row["val_pointing"] = loc["val"].get("macro_pointing_hit_rate")
            row["val_eib"] = loc["val"].get("macro_eib")
            macro = loc["val"].get("macro", {})
            if "eib_minus_prior" in macro:
                row["val_eib_minus_prior"] = macro["eib_minus_prior"]

        test = _read(run_dir / "cls_test" / "test_metrics.json")
        if test:
            row["test_macro_auroc"] = test.get("macro_auroc")
            row["test_macro_average_precision"] = test.get("macro_average_precision")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs):
    ok = runs[runs["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame()
    metrics = [c for c in ok.columns if c.startswith(("val_", "test_"))]
    grouped = ok.groupby("config")
    table = grouped[metrics].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table.insert(0, "n_seeds", grouped.size())
    table.insert(1, "seeds", grouped["seed"].apply(lambda s: ",".join(str(int(x)) for x in sorted(s.dropna()))))
    sort_key = "val_macro_average_precision_mean"
    return table.sort_values(sort_key, ascending=False) if sort_key in table else table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", help="Thư mục chứa các run (vd outputs/modal_runs)")
    parser.add_argument("--pattern", default="*", help="Glob tên run, vd 'loss_r50_*'")
    parser.add_argument("--output", default=None, help="Ghi bảng tổng hợp ra CSV")
    args = parser.parse_args()

    runs = collect(args.root, args.pattern)
    if runs.empty:
        raise SystemExit("Không tìm thấy run nào")
    incomplete = runs.loc[runs["status"] != "ok", "run"].tolist()
    if incomplete:
        print(f"Run chưa xong (bỏ qua): {incomplete}")

    table = summarize(runs)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.round(4).to_string())
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output)
        runs.to_csv(Path(args.output).with_name(Path(args.output).stem + "_runs.csv"), index=False)


if __name__ == "__main__":
    main()
