from __future__ import annotations

"""Plot repeated-training R2 histories for saved models."""

import argparse
import re
from pathlib import Path

from eesc.project import ANALYSIS_REPORT_DIR, TASKS, configure_matplotlib_env

configure_matplotlib_env()

import matplotlib.pyplot as plt
import numpy as np

RUN_PATTERN = re.compile(r"第 (?P<run>\d+) 次运行结束 \| RMSE: (?P<rmse>[-+]?\d*\.?\d+), R2: (?P<r2>[-+]?\d*\.?\d+)")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_REPORT_DIR,
        help="Directory used to store generated figures.",
    )


def discover_model_logs(base_path: Path, task_prefix: str) -> dict[str, Path]:
    model_logs = {}
    for log_path in sorted(base_path.glob(f"{task_prefix}_*_pth/training_history.log")):
        model_name = log_path.parent.name.replace(f"{task_prefix}_", "").replace("_pth", "")
        model_logs[model_name] = log_path
    return model_logs


def parse_training_log(log_path: Path) -> dict[str, object] | None:
    if not log_path.exists():
        print(f"Missing training history: {log_path}")
        return None

    runs = []
    r2_values = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = RUN_PATTERN.search(line)
        if not match:
            continue
        runs.append(int(match.group("run")))
        r2_values.append(float(match.group("r2")))

    if not runs:
        return None

    return {
        "runs": runs,
        "r2_values": r2_values,
        "max_r2": max(r2_values),
        "avg_r2": float(np.mean(r2_values)),
    }


def plot_task_training(task_name: str, model_results: dict[str, dict[str, object]], output_dir: Path) -> None:
    available = [(name, result) for name, result in model_results.items() if result]
    if not available:
        print(f"Skip {task_name}: no training history found.")
        return

    figure, axes = plt.subplots(len(available), 1, figsize=(12, 4 * len(available)))
    if len(available) == 1:
        axes = [axes]

    for axis, (model_name, result) in zip(axes, available):
        axis.plot(result["runs"], result["r2_values"], marker="o", label=model_name)
        axis.axhline(result["max_r2"], color="r", linestyle="--", label=f"max={result['max_r2']:.4f}")
        axis.axhline(result["avg_r2"], color="g", linestyle="--", label=f"avg={result['avg_r2']:.4f}")
        axis.set_xlabel("Run")
        axis.set_ylabel("R2")
        axis.set_title(f"{task_name.upper()} - {model_name}")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    output_path = output_dir / f"{task_name}_training_r2.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved training history figure: {output_path}")


def plot_all_models_comparison(all_results: dict[str, dict[str, dict[str, object]]], output_dir: Path) -> None:
    figure, axes = plt.subplots(len(TASKS), 1, figsize=(12, 15))
    if len(TASKS) == 1:
        axes = [axes]

    for axis, task_name in zip(axes, TASKS):
        model_results = all_results.get(task_name, {})
        available = {name: result for name, result in model_results.items() if result}
        if not available:
            axis.set_axis_off()
            continue

        max_runs = max(max(result["runs"]) for result in available.values())
        x_values = np.arange(1, max_runs + 1)

        for model_name, result in available.items():
            run_to_r2 = dict(zip(result["runs"], result["r2_values"]))
            full_series = [run_to_r2.get(index, np.nan) for index in x_values]
            axis.plot(x_values, full_series, marker="o", label=model_name)

        axis.set_xlabel("Run")
        axis.set_ylabel("R2")
        axis.set_title(f"{task_name.upper()} - model comparison")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    output_path = output_dir / "all_models_r2_comparison.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved cross-model comparison figure: {output_path}")


def run(output_dir: Path = ANALYSIS_REPORT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, dict[str, object]]] = {}
    for task_name, task_config in TASKS.items():
        print(f"Scanning training histories for task: {task_name}")
        model_logs = discover_model_logs(task_config.best_model_dir, task_name)
        model_results = {name: parse_training_log(log_path) for name, log_path in model_logs.items()}
        all_results[task_name] = model_results
        plot_task_training(task_name, model_results, output_dir)

    plot_all_models_comparison(all_results, output_dir)
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

