"""
RQ2: Location-aware loss có cải thiện localization không, và classification thay đổi ra sao?

So sánh paired từng cặp (baseline λ=0, run λ>0) cùng kiến trúc + seed trên external
test, với bootstrap theo ảnh (src/metrics/localization_stats.py):
  - localization bằng Grad-CAM (trùng đại lượng L_loc tối ưu) và occlusion (độc lập)
  - box-occlusion: logit giảm khi che trong/ngoài box (phép thử hành vi)
  - macro-AUROC từ test_predictions.csv
Thêm bảng theo finding (hiệu EIB và AUROC) để trả lời finding nào hưởng lợi.
"""

import pandas as pd

from src.analysis.rq_data import (
    classification_by_finding,
    markdown_table,
    localization_by_finding,
    localization_records,
    predictions_path,
)
from src.metrics.localization_stats import (
    load_predictions,
    paired_auroc_diff,
    paired_localization_diff,
)

RECORD_METRICS = {
    "gradcam": ("pointing_hit", "eib", "eib_over_area", "eib_minus_prior", "iou_mean"),
    "occlusion": ("pointing_hit", "eib", "eib_over_area", "eib_minus_prior", "iou_mean",
                  "drop_inside", "drop_outside"),
}


def _prediction_findings(path):
    columns = pd.read_csv(path, nrows=0).columns
    return [c.split("::", 1)[1] for c in columns if c.startswith("target::")]


def pairs(runs):
    """Ghép mỗi run λ>0 với baseline cùng model + seed."""
    base = runs[runs["lambda"] == 0].set_index(["model", "seed"])
    out = []
    for row in runs[runs["lambda"] > 0].to_dict("records"):
        key = (row["model"], row["seed"])
        if key in base.index:
            ref = base.loc[key]
            if isinstance(ref, pd.DataFrame):
                ref = ref.iloc[0]
            out.append((ref.to_dict(), row))
    return out


def paired_table(runs, findings=None, n_boot=1000, seed=0, split="test"):
    rows = []
    for base, guided in pairs(runs):
        context = {
            "arch": guided["arch"], "seed": guided["seed"], "lambda": guided["lambda"],
            "baseline": base["run"], "guided": guided["run"],
        }
        pred_a = predictions_path(base["path"], split)
        pred_b = predictions_path(guided["path"], split)
        if pred_a and pred_b:
            names = findings or _prediction_findings(pred_a)
            result = paired_auroc_diff(
                load_predictions(pred_a, names), load_predictions(pred_b, names),
                n_boot=n_boot, seed=seed,
            )
            rows.append(context | {"method": "classification"} | result)
        for method, metrics in RECORD_METRICS.items():
            rec_a = localization_records(base["path"], split, method)
            rec_b = localization_records(guided["path"], split, method)
            if rec_a is None or rec_b is None:
                continue
            for metric in metrics:
                if metric in rec_a.columns and metric in rec_b.columns:
                    result = paired_localization_diff(rec_a, rec_b, metric, n_boot=n_boot, seed=seed)
                    rows.append(context | {"method": method} | result)
    return pd.DataFrame(rows)


def per_finding_table(runs, split="test", method="gradcam"):
    rows = []
    for base, guided in pairs(runs):
        cls_a = classification_by_finding(base["path"], split)
        cls_b = classification_by_finding(guided["path"], split)
        loc_a = localization_by_finding(base["path"], split, method)
        loc_b = localization_by_finding(guided["path"], split, method)
        if any(x is None for x in (cls_a, cls_b, loc_a, loc_b)):
            continue
        a = cls_a.merge(loc_a, on="finding")
        b = cls_b.merge(loc_b, on="finding")
        merged = a.merge(b, on="finding", suffixes=("_base", "_guided"))
        for metric in ("auroc", "auprc", "pointing_hit_rate", "eib", "eib_minus_prior"):
            if f"{metric}_base" in merged and f"{metric}_guided" in merged:
                merged[f"delta_{metric}"] = merged[f"{metric}_guided"] - merged[f"{metric}_base"]
        merged["arch"], merged["seed"], merged["lambda"] = guided["arch"], guided["seed"], guided["lambda"]
        keep = ["arch", "seed", "lambda", "finding", "n_pairs_base"] + [
            c for c in merged.columns if c.startswith("delta_")
        ]
        rows.append(merged[[c for c in keep if c in merged.columns]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def to_markdown(paired, per_finding):
    lines = ["# RQ2 — tác động của location-aware loss (external test)", ""]
    if not paired.empty:
        cols = ["arch", "seed", "lambda", "method", "metric", "a", "b", "diff",
                "ci_low", "ci_high", "p_two_sided"]
        lines += [
            "## So sánh paired (B = λ>0, A = baseline cùng kiến trúc + seed; CI bootstrap theo ảnh)",
            "",
            "Grad-CAM lớp cuối trùng CAM mà L_loc tối ưu; kết luận về localization nên dựa "
            "thêm vào occlusion và box-occlusion (`drop_inside`, `drop_outside`). "
            "`eib_minus_prior` so với heatmap prior giải phẫu không nhìn ảnh.",
            "",
            markdown_table(paired[[c for c in cols if c in paired.columns]], index=False),
            "",
        ]
    if not per_finding.empty:
        summary = per_finding.groupby("finding")[[c for c in per_finding.columns if c.startswith("delta_")]].mean()
        lines += [
            "## Theo finding (trung bình qua các cặp, Grad-CAM)",
            "",
            markdown_table(
                summary.sort_values("delta_eib", ascending=False)
                if "delta_eib" in summary else summary
            ),
            "",
        ]
    return "\n".join(lines)
