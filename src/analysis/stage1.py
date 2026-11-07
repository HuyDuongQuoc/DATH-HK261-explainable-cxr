import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from src.utils.checksums import file_sha256


WEIGHTING_ORDER = ["raw", "sqrt", "sqrtcap5", "sqrtcap3"]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _precision_at_recall(target, probability, minimum_recall):
    precision, recall, _ = precision_recall_curve(target, probability)
    eligible = precision[recall >= minimum_recall]
    return float(np.max(eligible)) if eligible.size else 0.0


def _prediction_arrays(path, findings):
    table = pd.read_csv(path, dtype={"image_id": str})
    target_columns = [f"target::{finding}" for finding in findings]
    probability_columns = [f"probability::{finding}" for finding in findings]
    missing = set(target_columns + probability_columns) - set(table.columns)
    if missing:
        raise ValueError(f"Validation predictions are missing columns: {sorted(missing)}")
    if not table.image_id.is_unique:
        raise ValueError(f"Validation prediction image IDs are not unique: {path}")
    targets = table[target_columns].to_numpy(dtype=np.uint8)
    probabilities = table[probability_columns].to_numpy(dtype=float)
    if not np.isin(targets, [0, 1]).all():
        raise ValueError(f"Validation targets are not binary: {path}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Validation probabilities are not finite: {path}")
    return table.image_id.to_numpy(), targets, probabilities


def _telemetry_mean(path):
    if not path.is_file():
        return np.nan
    table = pd.read_csv(path)
    for column in ("utilization_gpu_percent", "utilization.gpu [%]"):
        if column in table:
            values = pd.to_numeric(table[column], errors="coerce")
            return float(values.mean())
    return np.nan


def _write_figures(macro, per_class, weights, histories, output_dir):
    figure_root = output_dir / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    if macro.empty:
        return figure_root

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=macro, x="model", y="macro_average_precision", hue="weighting", ax=ax)
    ax.set(title="Validation macro-AUPRC by positive weighting", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(figure_root / "01_macro_auprc_by_weighting.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=macro, x="macro_recall", y="macro_precision", hue="model", style="weighting", s=100, ax=ax)
    ax.set(title="Validation precision-recall trade-off", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout(); fig.savefig(figure_root / "02_precision_recall_tradeoff.png", dpi=200); plt.close(fig)

    delta_columns = ["delta_macro_average_precision_vs_raw", "delta_macro_precision_vs_raw", "delta_macro_recall_vs_raw"]
    delta_long = macro.melt(id_vars=["model", "weighting"], value_vars=delta_columns, var_name="metric", value_name="delta")
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=delta_long, x="model", y="delta", hue="metric", ax=ax)
    ax.axhline(0, color="black", linewidth=1); ax.set(title="Macro metric delta versus raw control")
    fig.tight_layout(); fig.savefig(figure_root / "03_delta_vs_raw.png", dpi=200); plt.close(fig)

    if not per_class.empty:
        matrix = per_class.pivot_table(index=["model", "finding"], columns="weighting", values="delta_average_precision_vs_raw")
        fig, ax = plt.subplots(figsize=(10, max(8, len(matrix) * 0.22)))
        sns.heatmap(matrix, cmap="RdBu_r", center=0, ax=ax)
        ax.set(title="Per-class AP delta versus raw")
        fig.tight_layout(); fig.savefig(figure_root / "04_per_class_ap_delta_heatmap.png", dpi=200); plt.close(fig)

    if not weights.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=weights, x="finding", y="effective_pos_weight", hue="weighting", style="model", markers=True, ax=ax)
        ax.tick_params(axis="x", rotation=75); ax.set(title="Effective positive weights")
        fig.tight_layout(); fig.savefig(figure_root / "05_effective_pos_weight_by_class.png", dpi=200); plt.close(fig)

    if not histories.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(data=histories, x="epoch", y="val_macro_ap", hue="model", style="weighting", ax=ax)
        ax.set(title="Validation macro-AUPRC training curves", ylim=(0, 1))
        fig.tight_layout(); fig.savefig(figure_root / "06_validation_auprc_training_curves.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=macro, x="model", y="train_hours", hue="weighting", ax=axes[0])
    sns.barplot(data=macro, x="model", y="mean_train_images_per_second", hue="weighting", ax=axes[1])
    axes[0].set_title("Training time"); axes[1].set_title("Training throughput")
    fig.tight_layout(); fig.savefig(figure_root / "07_training_time_throughput.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=macro, x="model", y="mean_gpu_utilization", hue="weighting", ax=axes[0])
    sns.barplot(data=macro, x="model", y="peak_gpu_memory_reserved_gb", hue="weighting", ax=axes[1])
    axes[0].set_title("Mean GPU utilization"); axes[1].set_title("Peak reserved GPU memory")
    fig.tight_layout(); fig.savefig(figure_root / "08_gpu_utilization_and_memory.png", dpi=200); plt.close(fig)
    return figure_root


def analyze_stage1(experiments, run_log_root, checkpoint_root, output_dir=None):
    run_log_root = Path(run_log_root)
    checkpoint_root = Path(checkpoint_root)
    output_dir = Path(output_dir or run_log_root / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    status_rows, macro_rows, class_rows, weight_rows, history_frames = [], [], [], [], []
    completed = {}
    for experiment in experiments:
        run_name = experiment["run_name"]
        result_dir = run_log_root / run_name
        checkpoint_path = checkpoint_root / run_name / "best.pt"
        paths = {
            "metrics": result_dir / "best_val_metrics.json",
            "predictions": result_dir / "best_val_predictions.csv",
            "loss": result_dir / "loss_metadata.json",
            "history": result_dir / "history.csv",
            "checkpoint": checkpoint_path,
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        status = "complete" if not missing else "pending"
        status_rows.append({**experiment, "status": status, "missing": ",".join(missing)})
        if missing:
            continue

        metrics = _read_json(paths["metrics"])
        loss = _read_json(paths["loss"])
        findings = [row["finding"] for row in metrics["per_class"]]
        ids, targets, probabilities = _prediction_arrays(paths["predictions"], findings)
        history = pd.read_csv(paths["history"])
        history["model"] = experiment["model_name"]
        history["weighting"] = experiment["weighting"]
        history_frames.append(history)
        telemetry = run_log_root / "hardware" / f"gpu_telemetry_{run_name}.csv"
        train_hours = float((history.train_seconds.sum() + history.val_seconds.sum()) / 3600)
        allocated = history.get("peak_gpu_memory_allocated_gb", history["peak_gpu_memory_gb"])
        reserved = history.get("peak_gpu_memory_reserved_gb", allocated)
        row = {
            "run_name": run_name,
            "model": experiment["model_name"],
            "weighting": experiment["weighting"],
            "status": "complete",
            "best_epoch": int(metrics["checkpoint_epoch"]),
            "macro_average_precision": float(metrics["macro_average_precision"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "macro_auroc": float(metrics["macro_auroc"]),
            "train_hours": train_hours,
            "mean_train_images_per_second": float(history.train_images_per_second.mean()),
            "mean_gpu_utilization": _telemetry_mean(telemetry),
            "peak_gpu_memory_allocated_gb": float(allocated.max()),
            "peak_gpu_memory_reserved_gb": float(reserved.max()),
            "checkpoint_sha256": file_sha256(checkpoint_path),
        }
        macro_rows.append(row)
        for metric_row in metrics["per_class"]:
            class_rows.append({"model": experiment["model_name"], "weighting": experiment["weighting"], **metric_row})
        for index, finding in enumerate(findings):
            weight_rows.append({
                "model": experiment["model_name"], "weighting": experiment["weighting"], "finding": finding,
                "positive_count": loss["positive_count"][index], "negative_count": loss["negative_count"][index],
                "raw_pos_weight": loss["raw_pos_weight"][index], "effective_pos_weight": loss["effective_pos_weight"][index],
            })
        completed[(experiment["model_name"], experiment["weighting"])] = {
            "metrics": metrics, "ids": ids, "targets": targets, "probabilities": probabilities, "macro": row,
        }

    status = pd.DataFrame(status_rows)
    macro = pd.DataFrame(macro_rows)
    per_class = pd.DataFrame(class_rows)
    weights = pd.DataFrame(weight_rows)
    histories = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    ranking_rows, selection = [], {}

    if not macro.empty:
        for column in (
            "precision_at_control_recall",
            "classes_ap_improved",
            "classes_ap_decreased",
            "classes_ap_drop_gt_002",
            "delta_macro_average_precision_vs_raw",
            "delta_macro_precision_vs_raw",
            "delta_macro_recall_vs_raw",
        ):
            macro[column] = np.nan
        for model in macro.model.unique():
            raw_key = (model, "raw")
            if raw_key not in completed:
                continue
            raw = completed[raw_key]
            raw_classes = {row["finding"]: row for row in raw["metrics"]["per_class"]}
            reference_ids, reference_targets = raw["ids"], raw["targets"]
            for weighting in macro.loc[macro.model == model, "weighting"]:
                item = completed[(model, weighting)]
                if not np.array_equal(item["ids"], reference_ids) or not np.array_equal(item["targets"], reference_targets):
                    raise ValueError(f"Validation predictions are not aligned for {model}/{weighting}")
                precisions = []
                deltas = []
                for index, finding in enumerate(raw_classes):
                    precisions.append(_precision_at_recall(
                        item["targets"][:, index], item["probabilities"][:, index], raw_classes[finding]["recall"]
                    ))
                    current_class = next(row for row in item["metrics"]["per_class"] if row["finding"] == finding)
                    deltas.append(current_class["average_precision"] - raw_classes[finding]["average_precision"])
                mask = (macro.model == model) & (macro.weighting == weighting)
                macro.loc[mask, "precision_at_control_recall"] = float(np.mean(precisions))
                macro.loc[mask, "classes_ap_improved"] = int(np.sum(np.asarray(deltas) > 0))
                macro.loc[mask, "classes_ap_decreased"] = int(np.sum(np.asarray(deltas) < 0))
                macro.loc[mask, "classes_ap_drop_gt_002"] = int(np.sum(np.asarray(deltas) < -0.02))
            raw_macro = macro[(macro.model == model) & (macro.weighting == "raw")].iloc[0]
            for column in ("macro_average_precision", "macro_precision", "macro_recall"):
                macro.loc[macro.model == model, f"delta_{column}_vs_raw"] = macro.loc[macro.model == model, column] - raw_macro[column]

            model_rows = macro[macro.model == model].copy()
            model_rows["eligible"] = (
                (model_rows.macro_average_precision >= raw_macro.macro_average_precision - 0.005)
                & (model_rows.macro_precision >= raw_macro.macro_precision - 0.01)
                & (model_rows.macro_recall >= raw_macro.macro_recall - 0.03)
                & (model_rows.classes_ap_drop_gt_002 < 4)
            )
            eligible = model_rows[model_rows.eligible].copy()
            eligible["weight_order"] = eligible.weighting.map({name: i for i, name in enumerate(WEIGHTING_ORDER)})
            eligible = eligible.sort_values(
                ["macro_average_precision", "precision_at_control_recall", "macro_recall", "classes_ap_improved", "weight_order"],
                ascending=[False, False, False, False, True],
            )
            if len(eligible) >= 2:
                best_ap = float(eligible.macro_average_precision.max())
                near_mask = best_ap - eligible.macro_average_precision <= 0.003
                top = eligible[near_mask].sort_values(
                    ["precision_at_control_recall", "macro_recall", "classes_ap_improved", "weight_order"],
                    ascending=[False, False, False, True],
                )
                eligible = pd.concat([top, eligible[~near_mask]], ignore_index=True)
            for rank, (_, ranked) in enumerate(eligible.iterrows(), start=1):
                ranking_rows.append({**ranked.to_dict(), "rank": rank})
            if set(model_rows.weighting) == set(WEIGHTING_ORDER) and len(eligible):
                winner = eligible.iloc[0]
                runner_up = eligible.iloc[1] if len(eligible) > 1 else None
                selection[model] = {
                    "winner": winner.weighting,
                    "winner_run_name": winner.run_name,
                    "winner_checkpoint_sha256": winner.checkpoint_sha256,
                    "runner_up": None if runner_up is None else runner_up.weighting,
                    "macro_average_precision": float(winner.macro_average_precision),
                    "precision_at_control_recall": float(winner.precision_at_control_recall),
                }

        if not per_class.empty:
            raw_lookup = per_class[per_class.weighting == "raw"].set_index(["model", "finding"])
            for metric in ("average_precision", "precision", "recall"):
                values = []
                for _, row in per_class.iterrows():
                    key = (row.model, row.finding)
                    values.append(
                        row[metric] - raw_lookup.loc[key, metric]
                        if key in raw_lookup.index else np.nan
                    )
                per_class[f"delta_{metric}_vs_raw"] = values

    ranking = pd.DataFrame(ranking_rows)
    status.to_csv(output_dir / "stage1_run_status.csv", index=False)
    macro.to_csv(output_dir / "stage1_macro_comparison.csv", index=False)
    per_class.to_csv(output_dir / "stage1_per_class_comparison.csv", index=False)
    per_class.to_csv(output_dir / "stage1_delta_vs_raw.csv", index=False)
    weights.to_csv(output_dir / "stage1_weight_table.csv", index=False)
    ranking.to_csv(output_dir / "stage1_ranking.csv", index=False)
    (output_dir / "stage1_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    analysis = {"completed_runs": int((status.status == "complete").sum()), "total_runs": len(status), "selection": selection, "external_test_used": False}
    (output_dir / "stage1_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    lines = ["# Stage 1 positive-weight analysis", "", f"Completed: {analysis['completed_runs']}/{analysis['total_runs']} runs.", "", "External test used: **No**.", ""]
    for model, choice in selection.items():
        lines.extend([f"## {model}", "", f"Winner: `{choice['winner']}`; runner-up: `{choice['runner_up']}`.", f"Validation macro-AUPRC: {choice['macro_average_precision']:.4f}.", ""])
        if not per_class.empty:
            selected_classes = per_class[(per_class.model == model) & (per_class.weighting == choice["winner"])].sort_values(
                "delta_average_precision_vs_raw"
            )
            decreases = selected_classes.head(3)
            increases = selected_classes.tail(3).sort_values("delta_average_precision_vs_raw", ascending=False)
            lines.append("Largest AP gains: " + ", ".join(
                f"{row.finding} ({row.delta_average_precision_vs_raw:+.3f})" for _, row in increases.iterrows()
            ) + ".")
            lines.append("Largest AP declines: " + ", ".join(
                f"{row.finding} ({row.delta_average_precision_vs_raw:+.3f})" for _, row in decreases.iterrows()
            ) + ".")
            extreme = selected_classes[(selected_classes.threshold >= 0.99) | (selected_classes.threshold <= 0.01)]
            if len(extreme):
                lines.append("Extreme tuned thresholds: " + ", ".join(extreme.finding.astype(str)) + ".")
            lines.append("")
    pending_models = sorted(set(status.model_name) - set(selection)) if not status.empty else []
    if pending_models:
        lines.extend(["## Pending", "", "No winner is published yet for: " + ", ".join(pending_models) + ".", ""])
    (output_dir / "stage1_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_figures(macro, per_class, weights, histories, output_dir)
    return {"status": status, "macro": macro, "per_class": per_class, "weights": weights, "ranking": ranking, "selection": selection}
