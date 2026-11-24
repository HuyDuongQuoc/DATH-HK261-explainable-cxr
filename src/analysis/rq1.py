"""
RQ1: Classification tốt hơn có kéo theo localization tốt hơn không?

Ba góc nhìn, đều trên bảng run × finding:

1. Architecture-level: macro classification vs macro localization của các baseline
   (λ=0), mean ± sd qua seed, kèm thứ hạng.
2. Within-finding: với mỗi finding, Spearman giữa metric classification và
   localization qua các run baseline (khác kiến trúc/seed), rồi trung bình qua
   finding. Loại bỏ yếu tố "finding dễ thì cả hai đều cao". CI bootstrap theo finding.
   Pooled Spearman (mọi cặp run × finding) cũng được báo cáo nhưng bị chi phối
   bởi độ khó của finding.
3. Can thiệp: so mỗi run λ>0 với baseline cùng kiến trúc + seed. Nếu localization
   đổi nhiều mà classification gần như đứng yên thì hai trục tách rời.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.rq_data import markdown_table

CLS_METRICS = ("auroc", "auprc")
LOC_METRICS = ("pointing_hit_rate", "eib", "eib_over_area", "eib_minus_prior", "iou_mean")


def _available(table, metrics):
    return [m for m in metrics if m in table.columns and table[m].notna().any()]


def architecture_table(table):
    base = table[table["lambda"] == 0]
    metrics = _available(base, CLS_METRICS + LOC_METRICS)
    per_run = base.groupby(["arch", "run"])[metrics].mean()  # macro theo finding
    summary = per_run.groupby("arch").agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "n_seeds", per_run.groupby("arch").size())
    for metric in metrics:
        summary[f"{metric}_rank"] = summary[f"{metric}_mean"].rank(ascending=False, method="min")
    return summary


def _spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
        return np.nan
    return float(spearmanr(x[mask], y[mask]).statistic)


def correlation_table(table, n_boot=2000, seed=0, baseline_only=True):
    data = table[table["lambda"] == 0] if baseline_only else table
    rng = np.random.default_rng(seed)
    findings = data["finding"].unique()
    rows = []
    for cls_metric in _available(data, CLS_METRICS):
        for loc_metric in _available(data, LOC_METRICS):
            per_finding = {}
            for finding, group in data.groupby("finding"):
                per_finding[finding] = _spearman(
                    group[cls_metric].to_numpy(float), group[loc_metric].to_numpy(float)
                )
            values = np.array([per_finding[f] for f in findings], dtype=float)
            within = np.nanmean(values) if np.isfinite(values).any() else np.nan
            boots = []
            for _ in range(n_boot):
                sample = rng.choice(values, size=len(values), replace=True)
                if np.isfinite(sample).any():
                    boots.append(np.nanmean(sample))
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
            pooled = _spearman(data[cls_metric].to_numpy(float), data[loc_metric].to_numpy(float))
            rows.append({
                "cls_metric": cls_metric,
                "loc_metric": loc_metric,
                "within_finding_spearman": within,
                "within_ci_low": lo,
                "within_ci_high": hi,
                "n_findings": int(np.isfinite(values).sum()),
                "runs_per_finding": int(data.groupby("finding")["run"].nunique().median()),
                "pooled_spearman": pooled,
                "n_units": int(len(data)),
            })
    return pd.DataFrame(rows)


def intervention_table(table):
    """Hiệu (λ>0 trừ baseline cùng arch + seed) của metric macro theo finding."""
    metrics = _available(table, CLS_METRICS + LOC_METRICS)
    macro = table.groupby(["arch", "seed", "lambda", "run"])[metrics].mean().reset_index()
    base = macro[macro["lambda"] == 0].set_index(["arch", "seed"])
    rows = []
    for row in macro[macro["lambda"] > 0].to_dict("records"):
        key = (row["arch"], row["seed"])
        if key not in base.index:
            continue
        ref = base.loc[key]
        if isinstance(ref, pd.DataFrame):
            ref = ref.iloc[0]
        out = {"arch": row["arch"], "seed": row["seed"], "lambda": row["lambda"],
               "run": row["run"], "baseline_run": ref["run"]}
        for metric in metrics:
            out[f"delta_{metric}"] = row[metric] - ref[metric]
        rows.append(out)
    return pd.DataFrame(rows)


def to_markdown(arch, corr, intervention, split, method):
    lines = [f"# RQ1 — classification vs localization ({split}, {method})", ""]
    lines += ["## Theo kiến trúc (baseline λ=0, macro theo finding)", "", markdown_table(arch), ""]
    lines += [
        "## Tương quan Spearman",
        "",
        "`within_finding_spearman`: trung bình qua finding của Spearman giữa các run "
        "baseline trong cùng finding (CI bootstrap theo finding). `pooled_spearman` gộp "
        "mọi cặp run × finding nên bị chi phối bởi độ khó của finding.",
        "",
        markdown_table(corr, index=False, digits=3),
        "",
    ]
    if not intervention.empty:
        lines += [
            "## Can thiệp bằng L_loc (λ>0 trừ baseline cùng kiến trúc + seed)",
            "",
            markdown_table(intervention, index=False),
            "",
        ]
    return "\n".join(lines)
