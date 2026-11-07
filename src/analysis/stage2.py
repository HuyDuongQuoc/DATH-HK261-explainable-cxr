import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.stage1 import (
    _prediction_arrays,
    _precision_at_recall,
    _telemetry_mean,
)
from src.utils.checksums import file_sha256


VARIANT_ORDER = ["bce", "focal_g05", "focal_g10", "focal_g15"]
WATCH_FINDINGS = {
    "densenet121": ["Pneumothorax", "Consolidation", "Atelectasis"],
    "resnet50": [],
    "vit_b_16": [],
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stable_validation_peak(history, tolerance=0.01):
    if "val_macro_ap" not in history or len(history) < 2:
        return False
    values = pd.to_numeric(history["val_macro_ap"], errors="coerce").to_numpy()
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return False
    return bool(np.sum(values >= np.max(values) - tolerance) >= 2)


def _write_figures(macro, per_class, histories, output_dir):
    figure_root = output_dir / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    if macro.empty:
        return
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=macro, x="variant", y="macro_average_precision", ax=ax)
    ax.set(title="Stage 2 validation macro-AUPRC", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(figure_root / "01_macro_auprc.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=macro, x="macro_recall", y="macro_precision", hue="variant", s=110, ax=ax)
    ax.set(title="Precision-recall trade-off", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout(); fig.savefig(figure_root / "02_precision_recall_tradeoff.png", dpi=200); plt.close(fig)

    delta_columns = [
        "delta_macro_average_precision_vs_bce",
        "delta_macro_precision_vs_bce",
        "delta_macro_recall_vs_bce",
        "delta_macro_f1_vs_bce",
    ]
    delta = macro.melt(id_vars=["variant"], value_vars=delta_columns, var_name="metric", value_name="delta")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=delta, x="variant", y="delta", hue="metric", ax=ax)
    ax.axhline(0, color="black", linewidth=1); ax.set(title="Delta versus Stage 2 BCE control")
    fig.tight_layout(); fig.savefig(figure_root / "03_delta_vs_bce.png", dpi=200); plt.close(fig)

    if not per_class.empty:
        matrix = per_class.pivot_table(index="finding", columns="variant", values="delta_average_precision_vs_bce")
        fig, ax = plt.subplots(figsize=(9, max(6, len(matrix) * 0.35)))
        sns.heatmap(matrix, cmap="RdBu_r", center=0, ax=ax)
        ax.set(title="Per-class AP delta versus BCE")
        fig.tight_layout(); fig.savefig(figure_root / "04_per_class_ap_delta_heatmap.png", dpi=200); plt.close(fig)

    if not histories.empty:
        fig, ax = plt.subplots(figsize=(11, 7))
        sns.lineplot(data=histories, x="epoch", y="val_macro_ap", hue="variant", ax=ax)
        ax.set(title="Validation macro-AUPRC training curves", ylim=(0, 1))
        fig.tight_layout(); fig.savefig(figure_root / "05_training_curves.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.barplot(data=macro, x="variant", y="train_hours", ax=axes[0])
    sns.barplot(data=macro, x="variant", y="mean_train_images_per_second", ax=axes[1])
    sns.barplot(data=macro, x="variant", y="mean_gpu_utilization", ax=axes[2])
    axes[0].set_title("Training time"); axes[1].set_title("Throughput"); axes[2].set_title("GPU utilization")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(figure_root / "06_performance_and_gpu.png", dpi=200); plt.close(fig)


def analyze_stage2(experiments, run_log_root, checkpoint_root, output_dir=None):
    """Build a model-local Stage 2 report without touching external-test data."""
    run_log_root = Path(run_log_root)
    checkpoint_root = Path(checkpoint_root)
    output_dir = Path(output_dir or run_log_root / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    status_rows, macro_rows, class_rows, history_frames = [], [], [], []
    completed = {}
    for experiment in experiments:
        run_name = experiment["run_name"]
        result_dir = run_log_root / "runs" / run_name
        checkpoint_path = checkpoint_root / run_name / "best.pt"
        paths = {
            "metrics": result_dir / "best_val_metrics.json",
            "predictions": result_dir / "best_val_predictions.csv",
            "loss": result_dir / "loss_metadata.json",
            "history": result_dir / "history.csv",
            "checkpoint": checkpoint_path,
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        status_rows.append({**experiment, "status": "pending" if missing else "complete", "missing": ",".join(missing)})
        if missing:
            continue

        metrics = _read_json(paths["metrics"])
        findings = [row["finding"] for row in metrics["per_class"]]
        ids, targets, probabilities = _prediction_arrays(paths["predictions"], findings)
        history = pd.read_csv(paths["history"])
        history["variant"] = experiment["variant"]
        history_frames.append(history)
        allocated = history.get("peak_gpu_memory_allocated_gb", history["peak_gpu_memory_gb"])
        reserved = history.get("peak_gpu_memory_reserved_gb", allocated)
        telemetry = run_log_root / "hardware" / f"gpu_telemetry_{run_name}.csv"
        row = {
            "run_name": run_name,
            "model": experiment["model_name"],
            "variant": experiment["variant"],
            "loss_name": experiment["loss_name"],
            "gamma": experiment.get("gamma"),
            "weighting": experiment["weighting"],
            "status": "complete",
            "best_epoch": int(metrics["checkpoint_epoch"]),
            "macro_average_precision": float(metrics["macro_average_precision"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "macro_auroc": float(metrics["macro_auroc"]),
            "stable_validation_peak": _stable_validation_peak(history),
            "train_hours": float((history.train_seconds.sum() + history.val_seconds.sum()) / 3600),
            "mean_train_images_per_second": float(history.train_images_per_second.mean()),
            "mean_gpu_utilization": _telemetry_mean(telemetry),
            "peak_gpu_memory_allocated_gb": float(allocated.max()),
            "peak_gpu_memory_reserved_gb": float(reserved.max()),
            "checkpoint_sha256": file_sha256(checkpoint_path),
        }
        macro_rows.append(row)
        for metric_row in metrics["per_class"]:
            class_rows.append({"model": experiment["model_name"], "variant": experiment["variant"], **metric_row})
        completed[experiment["variant"]] = {
            "metrics": metrics,
            "ids": ids,
            "targets": targets,
            "probabilities": probabilities,
        }

    status = pd.DataFrame(status_rows)
    macro = pd.DataFrame(macro_rows)
    per_class = pd.DataFrame(class_rows)
    histories = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    selection = {}

    comparison_columns = [
        "precision_at_control_recall", "classes_ap_improved", "classes_ap_decreased",
        "classes_ap_drop_gt_002", "delta_macro_average_precision_vs_bce",
        "delta_macro_precision_vs_bce", "delta_macro_recall_vs_bce", "delta_macro_f1_vs_bce",
        "eligible",
    ]
    for column in comparison_columns:
        macro[column] = False if column == "eligible" else np.nan

    if "bce" in completed:
        control = completed["bce"]
        control_classes = {row["finding"]: row for row in control["metrics"]["per_class"]}
        for variant, item in completed.items():
            if not np.array_equal(item["ids"], control["ids"]) or not np.array_equal(item["targets"], control["targets"]):
                raise ValueError(f"Validation predictions are not aligned for {variant}")
            precisions, ap_deltas = [], []
            for index, finding in enumerate(control_classes):
                precisions.append(_precision_at_recall(
                    item["targets"][:, index], item["probabilities"][:, index], control_classes[finding]["recall"]
                ))
                current = next(row for row in item["metrics"]["per_class"] if row["finding"] == finding)
                ap_deltas.append(float(current["average_precision"] - control_classes[finding]["average_precision"]))
            mask = macro.variant == variant
            macro.loc[mask, "precision_at_control_recall"] = float(np.mean(precisions))
            macro.loc[mask, "classes_ap_improved"] = int(np.sum(np.asarray(ap_deltas) > 0))
            macro.loc[mask, "classes_ap_decreased"] = int(np.sum(np.asarray(ap_deltas) < 0))
            macro.loc[mask, "classes_ap_drop_gt_002"] = int(np.sum(np.asarray(ap_deltas) < -0.02))

        control_row = macro[macro.variant == "bce"].iloc[0]
        for metric in ("macro_average_precision", "macro_precision", "macro_recall", "macro_f1"):
            macro[f"delta_{metric}_vs_bce"] = macro[metric] - float(control_row[metric])
        macro.loc[macro.variant == "bce", "eligible"] = True
        focal_mask = macro.loss_name == "sigmoid_focal"
        macro.loc[focal_mask, "eligible"] = (
            (macro.loc[focal_mask, "macro_average_precision"] >= float(control_row.macro_average_precision) + 0.005)
            & (macro.loc[focal_mask, "precision_at_control_recall"] >= float(control_row.precision_at_control_recall))
            & (macro.loc[focal_mask, "macro_recall"] >= float(control_row.macro_recall) - 0.03)
            & (macro.loc[focal_mask, "classes_ap_drop_gt_002"] < 4)
            & macro.loc[focal_mask, "stable_validation_peak"].astype(bool)
        )

        if not per_class.empty:
            lookup = per_class[per_class.variant == "bce"].set_index("finding")
            for metric in ("average_precision", "precision", "recall", "f1"):
                per_class[f"delta_{metric}_vs_bce"] = [
                    row[metric] - lookup.loc[row.finding, metric] for _, row in per_class.iterrows()
                ]

    ranking = macro.copy()
    if not ranking.empty:
        ranking["variant_order"] = ranking.variant.map({name: i for i, name in enumerate(VARIANT_ORDER)})
        eligible_focal = ranking[(ranking.loss_name == "sigmoid_focal") & (ranking.eligible == True)].copy()
        if eligible_focal.empty:
            ordered = pd.concat([ranking[ranking.variant == "bce"], ranking[ranking.variant != "bce"]])
        else:
            best_ap = float(eligible_focal.macro_average_precision.max())
            near = eligible_focal[best_ap - eligible_focal.macro_average_precision <= 0.003].sort_values(
                ["precision_at_control_recall", "macro_recall", "classes_ap_drop_gt_002", "gamma"],
                ascending=[False, False, True, True],
            )
            remaining = eligible_focal.drop(index=near.index).sort_values("macro_average_precision", ascending=False)
            ineligible = ranking[~ranking.index.isin(eligible_focal.index) & (ranking.variant != "bce")]
            ordered = pd.concat([near, remaining, ranking[ranking.variant == "bce"], ineligible])
        ranking = ordered.reset_index(drop=True)
        ranking["rank"] = np.arange(1, len(ranking) + 1)

    all_complete = len(status) == len(VARIANT_ORDER) and bool((status.status == "complete").all())
    if all_complete and not ranking.empty and "bce" in completed:
        winner = ranking.iloc[0]
        runner_up = ranking.iloc[1]
        selection = {
            "model": str(winner.model),
            "winner": str(winner.variant),
            "winner_run_name": str(winner.run_name),
            "winner_checkpoint_sha256": str(winner.checkpoint_sha256),
            "runner_up": str(runner_up.variant),
            "macro_average_precision": float(winner.macro_average_precision),
            "precision_at_control_recall": float(winner.precision_at_control_recall),
            "focal_replaced_bce": bool(winner.loss_name == "sigmoid_focal"),
        }

    status.to_csv(output_dir / "stage2_run_status.csv", index=False)
    macro.to_csv(output_dir / "stage2_macro_comparison.csv", index=False)
    per_class.to_csv(output_dir / "stage2_per_class_comparison.csv", index=False)
    per_class.to_csv(output_dir / "stage2_delta_vs_bce.csv", index=False)
    ranking.to_csv(output_dir / "stage2_ranking.csv", index=False)
    (output_dir / "stage2_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    analysis = {
        "completed_runs": int((status.status == "complete").sum()),
        "total_runs": len(status),
        "selection": selection,
        "external_test_used": False,
    }
    (output_dir / "stage2_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    model = experiments[0]["model_name"] if experiments else "unknown"
    lines = [
        f"# Stage 2 focal-loss analysis — {model}", "",
        f"Completed: {analysis['completed_runs']}/{analysis['total_runs']} runs.", "",
        "External test used: **No**.", "",
    ]
    if selection:
        lines.extend([
            "## Selection", "",
            f"Winner: `{selection['winner']}`; runner-up: `{selection['runner_up']}`.",
            f"Validation macro-AUPRC: {selection['macro_average_precision']:.4f}.",
            f"Focal replaces BCE: **{'Yes' if selection['focal_replaced_bce'] else 'No'}**.", "",
        ])
    else:
        lines.extend(["## Selection", "", "No winner is published until all four runs are complete.", ""])
    if model == "densenet121":
        lines.append("Priority review: Pneumothorax, Consolidation, and Atelectasis.")
    elif model == "resnet50":
        lines.append("Priority review: possible double-weighting from raw pos_weight plus focal modulation.")
    elif model == "vit_b_16":
        lines.append("Reference is the Stage 2 BCE sqrt-cap-3 control, not the Stage 1 raw run.")
    lines.append("")
    (output_dir / "stage2_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_figures(macro, per_class, histories, output_dir)
    return {"status": status, "macro": macro, "per_class": per_class, "ranking": ranking, "selection": selection}
