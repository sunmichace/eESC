from __future__ import annotations

"""Shared runtime helpers for the legacy km stacking scripts."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from eesc.tasks.km.models import STACKING_MODEL_SPECS, TOTAL_INPUT_DIM
from eesc.tasks.km.shared import DATA_DIR, DEVICE, NpyDataset, TASK_KEY, fold_file_stem

BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
STACKING_FOLDS = range(5, 6)

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingRegressor


def fold_file_path(split: str, kind: str, fold: int, data_dir: str | Path = DATA_DIR) -> Path:
    return Path(data_dir) / fold_file_stem(fold=fold, split=split, kind=kind)


def checkpoint_path(model_name: str, task_key: str = TASK_KEY) -> Path:
    return Path(f"{task_key}_{model_name}_pth") / "best.pth"


def load_checkpoint_model(model_class, model_name: str, input_dim: int = TOTAL_INPUT_DIM, task_key: str = TASK_KEY):
    save_path = checkpoint_path(model_name=model_name, task_key=task_key)
    model = model_class(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model, save_path


def calculate_adjusted_r2(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def calculate_mape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


def calculate_all_metrics(y_true, y_pred, n_features: int) -> dict[str, float]:
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    metrics = {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }
    metrics["adj_r2"] = calculate_adjusted_r2(metrics["r2"], len(y_true), n_features)
    metrics["mse"] = mean_squared_error(y_true, y_pred)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mape"] = calculate_mape(y_true, y_pred)
    return metrics


def get_predictions_with_context(model, loader):
    model.eval()
    preds_list = []
    labels_list = []
    env_list = []

    with torch.no_grad():
        for batch in loader:
            feat = batch["feat"].to(DEVICE)
            label = batch["label"]
            preds = model(feat)
            env_feat = feat[:, -2:]

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(label.numpy())
            env_list.extend(env_feat.cpu().numpy())

    return np.array(preds_list), np.array(labels_list), np.array(env_list)


def build_meta_features(preds_list, env_features):
    base_preds = np.column_stack(preds_list)
    pred_mean = np.mean(base_preds, axis=1, keepdims=True)
    pred_std = np.std(base_preds, axis=1, keepdims=True)
    return np.hstack([base_preds, env_features, pred_mean, pred_std])


def build_xgb_meta_model():
    if HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
        )

    return GradientBoostingRegressor(random_state=42)


__all__ = [
    "DATA_DIR",
    "DEVICE",
    "NpyDataset",
    "TASK_KEY",
    "TOTAL_INPUT_DIM",
    "BATCH_SIZE",
    "DEFAULT_NUM_WORKERS",
    "STACKING_FOLDS",
    "STACKING_MODEL_SPECS",
    "HAS_XGB",
    "fold_file_path",
    "checkpoint_path",
    "load_checkpoint_model",
    "calculate_adjusted_r2",
    "calculate_mape",
    "calculate_all_metrics",
    "get_predictions_with_context",
    "build_meta_features",
    "build_xgb_meta_model",
]
