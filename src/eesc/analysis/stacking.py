from __future__ import annotations

"""Generate summary figures from stacking logs across tasks."""

import argparse
import re
from pathlib import Path
from statistics import mean

from eesc.project import ANALYSIS_REPORT_DIR, TASKS, configure_matplotlib_env

configure_matplotlib_env()

import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ("xgb", "ridge", "final")
MODEL_LABELS = {"xgb": "XGBoost", "ridge": "Ridge", "final": "Final"}
METRIC_ORDER = ("r2", "adj_r2", "mae", "mse", "rmse", "mape")
METRIC_LABELS = {
    "r2": "R2 Score",
    "adj_r2": "Adjusted R2",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "mape": "MAPE (%)",
}
METRIC_REGEX = (
    r"R2=(?P<r2>[-+]?\d*\.?\d+), "
    r"Adj R2=(?P<adj_r2>[-+]?\d*\.?\d+), "
    r"MAE=(?P<mae>[-+]?\d*\.?\d+), "
    r"MSE=(?P<mse>[-+]?\d*\.?\d+), "
    r"RMSE=(?P<rmse>[-+]?\d*\.?\d+), "
    r"MAPE=(?P<mape>[-+]?\d*\.?\d+)%"
)
FOLD_PATTERN = re.compile(r"\[Fold (?P<fold>\d+)\]")
MODEL_PATTERN = re.compile(rf"(?P<model>XGB|Ridge|Final): {METRIC_REGEX}")
AVERAGE_PATTERN = re.compile(
    rf"Average (?P<model>XGBoost|Ridge|Final):.*?{METRIC_REGEX}",
    re.DOTALL,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_REPORT_DIR,
        help="Directory used to store generated figures.",
    )


def normalize_model_name(name: str) -> str:
    lowered = name.lower()
    if lowered == "xgboost":
        return "xgb"
    return lowered


def parse_metric_match(match: re.Match[str]) -> dict[str, float]:
    return {metric: float(match.group(metric)) for metric in METRIC_ORDER}


def compute_average_metrics(folds: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for fold in folds:
        for model_name, metrics in fold["models"].items():
            grouped.setdefault(model_name, []).append(metrics)

    return {
        model_name: {
            metric: mean(metric_set[metric] for metric_set in metric_sets) for metric in METRIC_ORDER
        }
        for model_name, metric_sets in grouped.items()
    }


def parse_train_log(file_path: Path) -> dict[str, object] | None:
    if not file_path.exists():
        print(f"Missing log file: {file_path}")
        return None

    text = file_path.read_text(encoding="utf-8")
    results: dict[str, object] = {"folds": [], "averages": {}}
    current_fold: dict[str, object] | None = None

    for line in text.splitlines():
        fold_match = FOLD_PATTERN.search(line)
        if fold_match:
            current_fold = {"fold": int(fold_match.group("fold")), "models": {}}
            results["folds"].append(current_fold)
            continue

        model_match = MODEL_PATTERN.search(line)
        if model_match and current_fold is not None:
            model_name = normalize_model_name(model_match.group("model"))
            current_fold["models"][model_name] = parse_metric_match(model_match)

    averages: dict[str, dict[str, float]] = {}
    for average_match in AVERAGE_PATTERN.finditer(text):
        model_name = normalize_model_name(average_match.group("model"))
        averages[model_name] = parse_metric_match(average_match)

    if not averages:
        averages = compute_average_metrics(results["folds"])

    results["averages"] = averages
    return results


def available_models(results: dict[str, object]) -> list[str]:
    discovered = set()
    for fold in results["folds"]:
        discovered.update(fold["models"].keys())
    return [name for name in MODEL_ORDER if name in discovered]


def plot_task_performance(task_name: str, results: dict[str, object], output_dir: Path) -> None:
    models = available_models(results)
    if not models:
        print(f"Skip {task_name}: no fold-level metrics found.")
        return

    folds = [fold["fold"] for fold in results["folds"]]
    figure, axes = plt.subplots(3, 2, figsize=(16, 18))

    for axis, metric in zip(axes.flatten(), METRIC_ORDER):
        for model_name in models:
            values = [
                fold["models"].get(model_name, {}).get(metric, np.nan) for fold in results["folds"]
            ]
            axis.plot(folds, values, marker="o", label=MODEL_LABELS[model_name])

        axis.set_xlabel("Fold")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.set_title(f"{task_name.upper()} - {METRIC_LABELS[metric]}")
        axis.set_xticks(folds)
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    output_path = output_dir / f"{task_name}_performance.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved task performance figure: {output_path}")


def preferred_average(results: dict[str, object]) -> tuple[str, dict[str, float]] | tuple[None, None]:
    for model_name in ("final", "ridge", "xgb"):
        if model_name in results["averages"]:
            return model_name, results["averages"][model_name]
    return None, None


def plot_tasks_comparison(all_results: dict[str, dict[str, object]], output_dir: Path) -> None:
    rows = []
    for task_name in TASKS:
        result = all_results.get(task_name)
        if not result:
            continue

        model_name, metrics = preferred_average(result)
        if metrics:
            rows.append((task_name, model_name, metrics))

    if not rows:
        print("Skip cross-task comparison: no summary metrics found.")
        return

    labels = [f"{task.upper()}\n({MODEL_LABELS[model]})" for task, model, _ in rows]
    x_positions = np.arange(len(rows))
    figure, axes = plt.subplots(3, 2, figsize=(16, 18))

    for axis, metric in zip(axes.flatten(), METRIC_ORDER):
        values = [metrics[metric] for _, _, metrics in rows]
        axis.bar(x_positions, values, width=0.5, alpha=0.8)
        axis.set_xlabel("Task")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.set_title(f"Cross-task comparison - {METRIC_LABELS[metric]}")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels)
        axis.grid(True, alpha=0.3)

    figure.tight_layout()
    output_path = output_dir / "tasks_comparison.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved cross-task comparison figure: {output_path}")


def generate_summary_table(all_results: dict[str, dict[str, object]], output_dir: Path) -> None:
    rows = []
    for task_name in TASKS:
        result = all_results.get(task_name)
        if not result:
            continue

        model_name, metrics = preferred_average(result)
        if not metrics:
            continue

        rows.append(
            [
                task_name.upper(),
                MODEL_LABELS[model_name],
                f"{metrics['r2']:.4f}",
                f"{metrics['adj_r2']:.4f}",
                f"{metrics['mae']:.4f}",
                f"{metrics['mse']:.4f}",
                f"{metrics['rmse']:.4f}",
                f"{metrics['mape']:.4f}",
            ]
        )

    if not rows:
        print("Skip summary table: no summary metrics found.")
        return

    figure, axis = plt.subplots(figsize=(16, 6))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Task", "Summary Source", "R2", "Adjusted R2", "MAE", "MSE", "RMSE", "MAPE (%)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.5)
    axis.set_title("KM/Kcat task summary", fontsize=14, pad=16)

    output_path = output_dir / "performance_summary.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved summary table figure: {output_path}")


def run(output_dir: Path = ANALYSIS_REPORT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, object]] = {}
    for task_name, task_config in TASKS.items():
        print(f"Parsing task log: {task_name} -> {task_config.machine_log_path}")
        parsed = parse_train_log(task_config.machine_log_path)
        if not parsed:
            continue

        all_results[task_name] = parsed
        plot_task_performance(task_name, parsed, output_dir)

    plot_tasks_comparison(all_results, output_dir)
    generate_summary_table(all_results, output_dir)
    print(f"All figures written to: {output_dir}")


def run_from_args(args: argparse.Namespace) -> None:
    run(output_dir=args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()

