from __future__ import annotations

"""Standardized task views that sit on top of legacy experiment layouts."""

import argparse
from dataclasses import dataclass

from eesc import legacy
from eesc.project import TASKS


@dataclass(frozen=True)
class TaskEntryPoint:
    alias: str
    script_path: str
    category: str
    description: str
    notes: str = ""
    config_hints: tuple[str, ...] = ()
    recommended: bool = False


@dataclass(frozen=True)
class TaskView:
    task_key: str
    summary: str
    canonical_data_dir: str
    artifact_dir: str
    archived_snapshot_dir: str | None
    entrypoints: tuple[TaskEntryPoint, ...]

    @property
    def task(self):
        return TASKS[self.task_key]


TASK_VIEWS = {
    "kcat": TaskView(
        task_key="kcat",
        summary="The most complete single-task branch, with runnable training, tuning, and stacking scripts.",
        canonical_data_dir="code/kcat/",
        artifact_dir="code/best_model_save/",
        archived_snapshot_dir="code/best_model_save/kcat/",
        entrypoints=(
            TaskEntryPoint(
                alias="train-5fold",
                script_path="featers_train_fold_2_f_f.py",
                category="training",
                description="Run the 5-fold training loop defined in the shared legacy training file.",
                config_hints=(
                    "Edit DEFAULT_EPOCHS and DEFAULT_BATCH_SIZE near the top of the script to change the __main__ defaults.",
                    "Edit ACTIVE_TRAIN_FOLDS near the top of the script to change which folds run.",
                    "Shared kcat model definitions now live in src/eesc/tasks/kcat/models.py and are re-exported through tasks/kcat_tc/code/legacy_shared.py.",
                ),
                recommended=True,
            ),
            TaskEntryPoint(
                alias="tune-cnn",
                script_path="tune_hyperparams_CNNPredictor.py",
                category="tuning",
                description="Optuna tuning entry for the CNN predictor baseline.",
                config_hints=(
                    "Edit TUNING_FOLD, TUNING_EPOCHS, and TUNING_TRIALS near the top of the script.",
                    "The script does not expose a stable public CLI yet; most changes still require editing constants in-file.",
                    "The CNNPredictor class itself is now centralized in src/eesc/tasks/kcat/models.py.",
                ),
                recommended=True,
            ),
            TaskEntryPoint(
                alias="tune-attention",
                script_path="tune_hyperparams_AttentionDTIModel.py",
                category="tuning",
                description="Optuna tuning entry for the attention-based model.",
                config_hints=(
                    "Edit TUNING_FOLD, TUNING_EPOCHS, and TUNING_TRIALS near the top of the script.",
                    "This alias now uses the shared AttentionDTITuningModel implementation in src/eesc/tasks/kcat/models.py.",
                ),
            ),
            TaskEntryPoint(
                alias="tune-dual-transformer",
                script_path="tune_hyperparams_DualTransformerFusionModel.py",
                category="tuning",
                description="Optuna tuning entry for the dual-transformer fusion model.",
                config_hints=(
                    "Edit TUNING_FOLD, TUNING_EPOCHS, and TUNING_TRIALS near the top of the script.",
                    "The DualTransformerFusionModel class itself is now centralized in src/eesc/tasks/kcat/models.py.",
                ),
            ),
            TaskEntryPoint(
                alias="tune-improved",
                script_path="tune_hyperparams_ImprovedPredictorModel.py",
                category="tuning",
                description="Optuna tuning entry for the improved predictor MLP.",
                config_hints=(
                    "Edit TUNING_FOLD, TUNING_EPOCHS, and TUNING_TRIALS near the top of the script.",
                    "The ImprovedPredictorModel class itself is now centralized in src/eesc/tasks/kcat/models.py.",
                ),
            ),
            TaskEntryPoint(
                alias="tune-gated",
                script_path="best_model_save/kcat_GatedInteractionModel_pth/tune_hyperparams_GatedInteractionModel.py",
                category="tuning",
                description="Archived Optuna tuning entry for the gated interaction model.",
                notes="Stored under best_model_save because the original script was saved alongside the winning checkpoint.",
                config_hints=(
                    "Edit TUNING_FOLD, TUNING_EPOCHS, and TUNING_TRIALS near the top of the archived script.",
                    "The GatedInteractionModel class itself is now centralized in src/eesc/tasks/kcat/models.py.",
                ),
            ),
            TaskEntryPoint(
                alias="stacking-xgb",
                script_path="best_model_save/essm_machine_xGboost_data.py",
                category="stacking",
                description="Train the task-level stacking/meta-model pipeline.",
                config_hints=(
                    "Reads saved checkpoints from best_model_save/kcat_*_pth/ and features from best_model_save/kcat/.",
                    "Edit STACKING_FOLDS, BATCH_SIZE, or DEFAULT_NUM_WORKERS in src/eesc/tasks/kcat/stacking_runtime.py to change the shared stacking defaults.",
                    "The base-model registry used by this script now lives in src/eesc/tasks/kcat/models.py as STACKING_MODEL_SPECS.",
                ),
                recommended=True,
            ),
            TaskEntryPoint(
                alias="stacking-merge",
                script_path="best_model_save/essm_machine_xGboost_data_merge.py",
                category="stacking",
                description="Variant of the stacking pipeline with merged meta-features.",
                config_hints=(
                    "Uses the merged/final-fusion stacking path under best_model_save/.",
                    "Edit STACKING_FOLDS, BATCH_SIZE, or DEFAULT_NUM_WORKERS in src/eesc/tasks/kcat/stacking_runtime.py to change the shared stacking defaults.",
                    "The base-model registry used by this script now lives in src/eesc/tasks/kcat/models.py as STACKING_MODEL_SPECS.",
                ),
            ),
            TaskEntryPoint(
                alias="auto-runner",
                script_path="auto_runner.py",
                category="automation",
                description="Repeat a training script and preserve the best checkpoint.",
                notes="The file still contains hard-coded script names and usually needs manual configuration before use.",
                config_hints=(
                    "Edit SCRIPT_NAME, SOURCE_WEIGHT, BEST_WEIGHT, LOG_FILE, MAX_RUNS, and OPTIMIZE_METRIC before execution.",
                ),
            ),
        ),
    ),
    "km": TaskView(
        task_key="km",
        summary="A stacking-focused task branch whose runnable legacy entries now share a standardized km package layer.",
        canonical_data_dir="code/best_model_save/km/",
        artifact_dir="code/best_model_save/",
        archived_snapshot_dir=None,
        entrypoints=(
            TaskEntryPoint(
                alias="stacking-xgb",
                script_path="best_model_save/essm_machine_xGboost_data.py",
                category="stacking",
                description="Train the saved Km stacking/meta-model pipeline.",
                config_hints=(
                    "Reads saved checkpoints from best_model_save/km_*_pth/ and features from best_model_save/km/preprocessed/.",
                    "Edit STACKING_FOLDS, BATCH_SIZE, or DEFAULT_NUM_WORKERS in src/eesc/tasks/km/stacking_runtime.py to change the shared stacking defaults.",
                    "The task-specific stacking model registry now lives in src/eesc/tasks/km/models.py as STACKING_MODEL_SPECS.",
                    "This base alias preserves the original XGBoost-only output path and writes machine_results/global_xgb_model.pkl.",
                ),
                recommended=True,
            ),
            TaskEntryPoint(
                alias="stacking-merge",
                script_path="best_model_save/essm_machine_xGboost_data_merge_machine.py",
                category="stacking",
                description="Merged-feature variant of the Km stacking pipeline.",
                config_hints=(
                    "Uses the merged/final-fusion stacking path under best_model_save/.",
                    "Edit STACKING_FOLDS, BATCH_SIZE, or DEFAULT_NUM_WORKERS in src/eesc/tasks/km/stacking_runtime.py to change the shared stacking defaults.",
                    "This alias preserves the original merged outputs machine_results/global_merge_xgb_model.pkl and machine_results/global_merge_ridge_model.pkl.",
                ),
            ),
        ),
    ),
    "kcatkm": TaskView(
        task_key="kcatkm",
        summary="A complete multi-model branch similar to kcat, now with its own standardized shared layer and runtime wrappers.",
        canonical_data_dir="code/kcatkm/",
        artifact_dir="code/best_model_save/",
        archived_snapshot_dir="code/best_model_save/kcatkm/",
        entrypoints=(
            TaskEntryPoint(
                alias="train-5fold",
                script_path="featers_train_fold_2_f_f.py",
                category="training",
                description="Run the 5-fold training loop for the joint/extended task branch.",
                config_hints=(
                    "Edit DEFAULT_EPOCHS and DEFAULT_BATCH_SIZE near the top of the script to change the __main__ defaults.",
                    "Edit ACTIVE_TRAIN_FOLDS near the top of the script to change which folds run.",
                    "The task-specific shared layer now lives under src/eesc/tasks/kcatkm/ and re-exports through tasks/kcatkm_tc/code/legacy_shared.py.",
                ),
                recommended=True,
            ),
            TaskEntryPoint(
                alias="tune-cnn",
                script_path="tune_hyperparams_CNNPredictor.py",
                category="tuning",
                description="Optuna tuning entry for the CNN predictor.",
                config_hints=(
                    "Edit TUNING_FOLD, TUNING_EPOCHS, and TUNING_TRIALS near the top of the script.",
                    "This task shares the generic CNN model implementation used by the standardized kcat modules.",
                ),
                recommended=True,
            ),
            TaskEntryPoint(
                alias="tune-attention",
                script_path="tune_hyperparams_AttentionDTIModel.py",
                category="tuning",
                description="Optuna tuning entry for the attention-based model.",
                config_hints=(
                    "Edit TUNING_FOLD, TUNING_EPOCHS, and TUNING_TRIALS near the top of the script.",
                    "This alias now uses the shared AttentionDTITuningModel implementation through src/eesc/tasks/kcatkm/.",
                ),
            ),
            TaskEntryPoint(
                alias="tune-dual-transformer",
                script_path="tune_hyperparams_DualTransformerFusionModel.py",
                category="tuning",
                description="Optuna tuning entry for the dual-transformer model.",
                config_hints=(
                    "Edit TUNING_FOLD, TUNING_EPOCHS, and TUNING_TRIALS near the top of the script.",
                    "This task shares the generic dual-transformer implementation used by the standardized kcat modules.",
                ),
            ),
            TaskEntryPoint(
                alias="tune-improved",
                script_path="tune_hyperparams_ImprovedPredictorModel.py",
                category="tuning",
                description="Optuna tuning entry for the improved predictor model.",
                config_hints=(
                    "Edit TUNING_FOLD, TUNING_EPOCHS, and TUNING_TRIALS near the top of the script.",
                    "This task shares the generic improved-predictor implementation used by the standardized kcat modules.",
                ),
            ),
            TaskEntryPoint(
                alias="tune-gated",
                script_path="tune_hyperparams_GatedInteractionModel.py",
                category="tuning",
                description="Optuna tuning entry for the gated interaction model.",
                config_hints=(
                    "Edit TUNING_FOLD, TUNING_EPOCHS, and TUNING_TRIALS near the top of the script.",
                    "This task shares the generic gated-interaction implementation used by the standardized kcat modules.",
                ),
            ),
            TaskEntryPoint(
                alias="stacking-xgb",
                script_path="best_model_save/essm_machine_xGboost_data.py",
                category="stacking",
                description="Train the task-level stacking/meta-model pipeline.",
                config_hints=(
                    "Reads saved checkpoints from best_model_save/kcatkm_*_pth/ and features from best_model_save/kcatkm/.",
                    "Edit STACKING_FOLDS, BATCH_SIZE, or DEFAULT_NUM_WORKERS in src/eesc/tasks/kcatkm/stacking_runtime.py to change the shared stacking defaults.",
                    "The task-specific stacking model registry now lives in src/eesc/tasks/kcatkm/models.py as STACKING_MODEL_SPECS.",
                ),
                recommended=True,
            ),
            TaskEntryPoint(
                alias="stacking-merge",
                script_path="best_model_save/essm_machine_xGboost_data_merge.py",
                category="stacking",
                description="Variant of the stacking pipeline with merged meta-features.",
                config_hints=(
                    "Uses the merged/final-fusion stacking path under best_model_save/.",
                    "Edit STACKING_FOLDS, BATCH_SIZE, or DEFAULT_NUM_WORKERS in src/eesc/tasks/kcatkm/stacking_runtime.py to change the shared stacking defaults.",
                    "The task-specific stacking model registry now lives in src/eesc/tasks/kcatkm/models.py as STACKING_MODEL_SPECS.",
                ),
            ),
            TaskEntryPoint(
                alias="auto-runner",
                script_path="auto_runner.py",
                category="automation",
                description="Repeat a training script and preserve the best checkpoint.",
                notes="The file still contains hard-coded script names and usually needs manual configuration before use.",
                config_hints=("Edit the hard-coded runner settings before using this entry.",),
            ),
        ),
    ),
}


def get_task_view(task_key: str) -> TaskView:
    return TASK_VIEWS[task_key]


def find_task_entrypoint(task_key: str, alias: str) -> TaskEntryPoint:
    task_view = get_task_view(task_key)
    for entrypoint in task_view.entrypoints:
        if entrypoint.alias == alias:
            return entrypoint
    available_aliases = ", ".join(entry.alias for entry in task_view.entrypoints)
    raise ValueError(f"Unknown task entrypoint '{alias}'. Available aliases: {available_aliases}")


def run_task_view(args: argparse.Namespace) -> None:
    task_view = get_task_view(args.task)
    print(f"[{task_view.task_key}] {task_view.summary}")
    print(f"canonical-data: {task_view.canonical_data_dir}")
    print(f"artifact-dir:    {task_view.artifact_dir}")
    if task_view.archived_snapshot_dir:
        print(f"archived-copy:   {task_view.archived_snapshot_dir}")
    print("entrypoints:")
    for entry in task_view.entrypoints:
        marker = "*" if entry.recommended else "-"
        print(
            f"  {marker} {entry.alias:<20} {entry.category:<10} "
            f"{entry.script_path} :: {entry.description}"
        )
        if entry.notes:
            print(f"    note: {entry.notes}")
        if entry.config_hints:
            print(f"    template: env PYTHONPATH=src python -m eesc.cli task-run {task_view.task_key} {entry.alias}")
            for hint in entry.config_hints:
                print(f"    config: {hint}")


def show_task_entrypoint_template(task_key: str, entrypoint: TaskEntryPoint) -> None:
    task_view = get_task_view(task_key)
    script_path = task_view.task.code_dir / entrypoint.script_path
    print(f"task: {task_key}")
    print(f"alias: {entrypoint.alias}")
    print(f"category: {entrypoint.category}")
    print(f"script: {entrypoint.script_path}")
    print(f"resolved-script-path: {script_path}")
    print(f"command: env PYTHONPATH=src python -m eesc.cli task-run {task_key} {entrypoint.alias}")
    print(f"with passthrough args: env PYTHONPATH=src python -m eesc.cli task-run {task_key} {entrypoint.alias} -- --arg value")
    print("defaults: most historical task entries still read defaults from constants inside the script rather than a stable public CLI.")
    if entrypoint.notes:
        print(f"note: {entrypoint.notes}")
    if entrypoint.config_hints:
        print("config hints:")
        for hint in entrypoint.config_hints:
            print(f"  - {hint}")


def run_task_entrypoint(args: argparse.Namespace) -> None:
    entrypoint = find_task_entrypoint(args.task, args.alias)
    if getattr(args, "show_template", False):
        show_task_entrypoint_template(args.task, entrypoint)
        return
    script = legacy.find_legacy_script(args.task, entrypoint.script_path, runnable_only=True)
    legacy.run_resolved_legacy_script(
        script,
        passthrough_args=list(getattr(args, "script_args", [])),
        dry_run=args.dry_run,
    )
