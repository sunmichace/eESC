from __future__ import annotations

"""Helpers for discovering and running legacy experiment scripts."""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from eesc.project import TASKS, TaskConfig


@dataclass(frozen=True)
class LegacyScript:
    task_key: str
    absolute_path: Path
    relative_to_code_dir: Path
    category: str
    description: str
    runnable: bool

    @property
    def task(self) -> TaskConfig:
        return TASKS[self.task_key]

    @property
    def working_dir(self) -> Path:
        return self.absolute_path.parent


def _model_name_from_tuning_script(script_name: str) -> str:
    stem = Path(script_name).stem
    return stem.removeprefix("tune_hyperparams_")


def _classify_legacy_script(task_key: str, absolute_path: Path) -> LegacyScript:
    task = TASKS[task_key]
    relative_path = absolute_path.relative_to(task.code_dir)
    file_name = relative_path.name

    if len(relative_path.parts) == 1:
        if file_name == "auto_runner.py":
            category = "automation"
            description = "Repeated training loop that keeps the best checkpoint."
            runnable = True
        elif file_name == "featers_train_fold_2_f_f.py":
            category = "training"
            description = "Legacy end-to-end training script with shared model definitions."
            runnable = True
        elif file_name.startswith("tune_hyperparams_"):
            category = "tuning"
            description = f"Optuna tuning entry for {_model_name_from_tuning_script(file_name)}."
            runnable = True
        else:
            category = "entry"
            description = "Legacy experiment entry script."
            runnable = True
    elif relative_path.parts[0] == "best_model_save" and len(relative_path.parts) == 2:
        if file_name.startswith("essm_machine_"):
            category = "stacking"
            description = "Stacking or meta-model training entry saved with best-model artifacts."
        else:
            category = "archived-entry"
            description = "Archived top-level experiment entry stored under best_model_save."
        runnable = True
    elif file_name.startswith("tune_hyperparams_"):
        category = "archived-tuning"
        description = f"Archived Optuna tuning entry for {_model_name_from_tuning_script(file_name)}."
        runnable = True
    else:
        category = "reference"
        description = "Archived model source stored alongside saved weights or logs."
        runnable = False

    return LegacyScript(
        task_key=task_key,
        absolute_path=absolute_path,
        relative_to_code_dir=relative_path,
        category=category,
        description=description,
        runnable=runnable,
    )


def discover_legacy_scripts(task_key: str) -> list[LegacyScript]:
    task = TASKS[task_key]
    scripts = []
    for absolute_path in sorted(task.code_dir.rglob("*.py")):
        if "__pycache__" in absolute_path.parts:
            continue
        scripts.append(_classify_legacy_script(task_key, absolute_path))
    return scripts


def list_runnable_legacy_scripts(task_key: str) -> list[LegacyScript]:
    return [script for script in discover_legacy_scripts(task_key) if script.runnable]


def find_legacy_script(task_key: str, script_selector: str, runnable_only: bool = True) -> LegacyScript:
    scripts = list_runnable_legacy_scripts(task_key) if runnable_only else discover_legacy_scripts(task_key)

    selector_path = Path(script_selector)
    exact_matches = [script for script in scripts if script.relative_to_code_dir == selector_path]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(f"Ambiguous script selector: {script_selector}")

    name_matches = [script for script in scripts if script.relative_to_code_dir.name == script_selector]
    if len(name_matches) == 1:
        return name_matches[0]
    if len(name_matches) > 1:
        candidates = ", ".join(script.relative_to_code_dir.as_posix() for script in name_matches)
        raise ValueError(f"Ambiguous script name '{script_selector}'. Use one of: {candidates}")

    available = ", ".join(script.relative_to_code_dir.as_posix() for script in scripts)
    raise ValueError(f"Unknown script '{script_selector}'. Available scripts: {available}")


def add_list_tasks_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print absolute task paths.",
    )


def run_list_tasks(args: argparse.Namespace) -> None:
    for task_key, task in TASKS.items():
        line = f"{task_key:<7} {task.directory_name:<12} {task.description}"
        print(line)
        if args.verbose:
            print(f"         root: {task.task_root}")


def run_resolved_legacy_script(
    script: LegacyScript,
    passthrough_args: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    passthrough_args = passthrough_args or []
    command = [sys.executable, script.absolute_path.name, *passthrough_args]

    if dry_run:
        print(f"task: {script.task_key}")
        print(f"cwd: {script.working_dir}")
        print(f"cmd: {' '.join(command)}")
        return

    result = subprocess.run(command, cwd=script.working_dir, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
