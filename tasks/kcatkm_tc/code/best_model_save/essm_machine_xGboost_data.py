from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from legacy_shared import DATA_DIR, TASK_KEY, TOTAL_INPUT_DIM
from eesc.tasks.kcatkm.stacking_runtime import (
    BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    STACKING_FOLDS,
    STACKING_MODEL_SPECS,
    NpyDataset,
    build_meta_features,
    calculate_all_metrics,
    get_predictions_with_context,
    load_checkpoint_model,
)

k_name = TASK_KEY
SAVE_DIR = Path("./machine_results")
LOG_FILE_PATH = SAVE_DIR / "train_log.txt"
RIDGE_MODEL_PATH = SAVE_DIR / "global_ridge_model.pkl"


def fold_file(split: str, kind: str, fold: int) -> Path:
    return Path(DATA_DIR) / f"{k_name}_fold_{fold}_{split}_{kind}.npy"


def build_loader(split: str, fold: int) -> DataLoader:
    dataset = NpyDataset(
        fold_file(split=split, kind="feat", fold=fold),
        fold_file(split=split, kind="label", fold=fold),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=DEFAULT_NUM_WORKERS)


def collect_fold_predictions(fold: int):
    val_loader = build_loader(split="val", fold=fold)
    test_loader = build_loader(split="test", fold=fold)

    fold_val_preds = []
    fold_test_preds = []
    fold_val_env = None
    fold_test_env = None
    current_fold_val_labels = None
    current_fold_test_labels = None

    for model_class, model_name in STACKING_MODEL_SPECS:
        model, save_path = load_checkpoint_model(model_class, model_name, input_dim=TOTAL_INPUT_DIM, task_key=TASK_KEY)
        print(f"  [Load] Loading pre-trained {model_name} for Fold {fold} From {save_path}...")

        pred_val, labels_val, env_val = get_predictions_with_context(model, val_loader)
        fold_val_preds.append(pred_val)
        if fold_val_env is None:
            fold_val_env = env_val
            current_fold_val_labels = labels_val

        pred_test, labels_test, env_test = get_predictions_with_context(model, test_loader)
        fold_test_preds.append(pred_test)
        if fold_test_env is None:
            fold_test_env = env_test
            current_fold_test_labels = labels_test

    x_val_meta = build_meta_features(fold_val_preds, fold_val_env)
    x_test_meta = build_meta_features(fold_test_preds, fold_test_env)
    return x_val_meta, current_fold_val_labels.ravel(), x_test_meta, current_fold_test_labels.ravel()


def run_global_stacking() -> None:
    SAVE_DIR.mkdir(exist_ok=True)
    original_stdout = sys.stdout

    with LOG_FILE_PATH.open("w", encoding="utf-8") as log_file:
        try:
            sys.stdout = log_file
            print(f"\n{'=' * 30} Start Global Stacking (All Folds) {'=' * 30}")

            all_val_meta_features = []
            all_val_labels = []
            fold_test_data = {}

            print(">>> Phase 1: Collecting Meta-Training Data from ALL 5 Folds...")
            for fold in STACKING_FOLDS:
                print(f"  Processing Fold {fold}...")
                x_val_meta, y_val, x_test_meta, y_test = collect_fold_predictions(fold)
                all_val_meta_features.append(x_val_meta)
                all_val_labels.append(y_val)
                fold_test_data[fold] = (x_test_meta, y_test)

            print("\n>>> Phase 2: Training Global Meta-Models...")
            x_meta_train_global = np.vstack(all_val_meta_features)
            y_meta_train_global = np.concatenate(all_val_labels)

            print(f"    Global Meta-Train Shape: {x_meta_train_global.shape}")
            print(f"    (Compared to single fold: {all_val_meta_features[0].shape})")

            ridge_model = Ridge(alpha=0.5)
            ridge_model.fit(x_meta_train_global, y_meta_train_global)

            print("\n>>> Phase 3: Final Evaluation on Test Sets...")
            global_results_ridge = []

            for fold in STACKING_FOLDS:
                x_test, y_test = fold_test_data[fold]
                n_features = x_test.shape[1]
                pred_ridge = ridge_model.predict(x_test)
                metrics_ridge = calculate_all_metrics(y_test, pred_ridge, n_features)
                global_results_ridge.append(metrics_ridge)

                print(f"  [Fold {fold}]")
                print(
                    "    Ridge: "
                    f"R2={metrics_ridge['r2']:.4f}, "
                    f"Adj R2={metrics_ridge['adj_r2']:.4f}, "
                    f"MAE={metrics_ridge['mae']:.4f}, "
                    f"MSE={metrics_ridge['mse']:.4f}, "
                    f"RMSE={metrics_ridge['rmse']:.4f}, "
                    f"MAPE={metrics_ridge['mape']:.4f}%"
                )

            print("\n" + "=" * 30)
            print("FINAL GLOBAL STACKING RESULTS")
            print("=" * 30)

            print("\n>>> Saving Models and Results...")
            joblib.dump(ridge_model, RIDGE_MODEL_PATH)
            print(f"  [Saved] Meta-models saved to {SAVE_DIR}")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    run_global_stacking()
