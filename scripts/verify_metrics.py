"""
Kiểm tra độc lập các chỉ số đã lưu cho mọi run (classification + localization).

    python scripts/verify_metrics.py outputs/modal_runs/volume

A. Tính lại AUROC/AP theo finding từ test_predictions.csv bằng sklearn, so với
   test_metrics.json; kiểm tra mọi run dùng cùng ảnh + nhãn; so với AUROC validation.
B. Kiểm tra records localization: miền giá trị, eib_over_area = eib/box_area,
   eib_minus_prior = eib - prior_eib, macro EIB khớp localization_summary.json,
   mốc ngẫu nhiên (pointing ≈ diện tích box) và prior giải phẫu; mọi run/method
   đánh giá trên cùng tập cặp (image, finding) dương.
Bỏ qua run theo tiền tố trong EXCLUDE (ViT train hỏng, ablation loss).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(sys.argv[1])
EXCLUDE = ("vit_base_", "vitgap_", "loss_")
runs = sorted(p for p in ROOT.iterdir()
              if (p / "cls_test" / "test_metrics.json").is_file() and not p.name.startswith(EXCLUDE))

print("=== A. Classification: tính lại từ test_predictions.csv ===")
ref_targets, ref_ids = None, None
rows = []
for run in runs:
    pred = pd.read_csv(run / "cls_test" / "test_predictions.csv")
    saved = json.loads((run / "cls_test" / "test_metrics.json").read_text())
    findings = [c.split("::", 1)[1] for c in pred.columns if c.startswith("target::")]
    T = pred[[f"target::{f}" for f in findings]].to_numpy()
    P = pred[[f"probability::{f}" for f in findings]].to_numpy()
    if ref_targets is None:
        ref_targets, ref_ids = T, pred["image_id"].tolist()
    same_labels = pred["image_id"].tolist() == ref_ids and np.array_equal(T, ref_targets)
    aucs = [roc_auc_score(T[:, i], P[:, i]) for i in range(len(findings))]
    aps = [average_precision_score(T[:, i], P[:, i]) for i in range(len(findings))]
    saved_auc = {c["finding"]: c["auroc"] for c in saved["per_class"]}
    max_err = max(abs(a - saved_auc[f]) for a, f in zip(aucs, findings))
    val = run / "res" / "best_val_metrics.json"
    val_auc = json.loads(val.read_text()).get("macro_auroc") if val.is_file() else np.nan
    rows.append({
        "run": run.name, "n": len(pred), "same_labels_as_first": same_labels,
        "macro_auroc": np.mean(aucs), "saved": saved["macro_auroc"], "max_abs_err": max_err,
        "macro_ap": np.mean(aps), "val_auroc": val_auc,
        "p_range": f"{P.min():.1e}..{P.max():.3f}", "nan": int(np.isnan(P).sum()),
    })
print(pd.DataFrame(rows).round(4).to_string(index=False))
prev = ref_targets.mean(0)
print("\nPrevalence test (AP ngẫu nhiên = prevalence), macro =", round(prev.mean(), 4))
print(pd.Series(prev, index=findings).round(4).to_string())

print("\n=== B. Localization records ===")
checks = []
pair_sets = {}
for run in runs:
    for method in ("gradcam", "occlusion"):
        f = run / f"loc_test_{method}" / "test_localization_records.csv"
        if not f.is_file():
            checks.append({"run": run.name, "method": method, "missing": True})
            continue
        r = pd.read_csv(f)
        key = frozenset(zip(r["image_id"], r["finding"]))
        pair_sets[(run.name, method)] = key
        summ = json.loads((run / f"loc_test_{method}" / "localization_summary.json").read_text())
        test_macro = summ.get("test", {}).get("macro", {})
        by_f = r.groupby("finding")
        checks.append({
            "run": run.name, "method": method, "n_pairs": len(r),
            "eib_in_[0,1]": bool(r["eib"].between(-1e-6, 1 + 1e-6).all()),
            "hit_binary": bool(r["pointing_hit"].isin([0, 1]).all()),
            "eib/area_ok": float(np.nanmax(np.abs(r["eib"] / r["box_area"] - r["eib_over_area"]))) < 1e-4,
            "minus_prior_ok": float(np.nanmax(np.abs(r["eib"] - r["prior_eib"] - r["eib_minus_prior"]))) < 1e-6,
            "macro_eib": by_f["eib"].mean().mean(),
            "saved_macro_eib": test_macro.get("eib", np.nan),
            "macro_hit": by_f["pointing_hit"].mean().mean(),
            "random_hit≈area": by_f["box_area"].mean().mean(),
            "prior_eib": by_f["prior_eib"].mean().mean(),
            "prior_hit": by_f["prior_hit"].mean().mean(),
            "nan_eib": int(r["eib"].isna().sum()),
        })
print(pd.DataFrame(checks).round(4).to_string(index=False))

sets = set(pair_sets.values())
print("\nSố tập cặp (image, finding) khác nhau giữa các run/method:", len(sets))
n_pos = int(ref_targets.sum())
print("Số cặp dương trong nhãn test:", n_pos, "| số cặp được đánh giá:", len(next(iter(sets))))
