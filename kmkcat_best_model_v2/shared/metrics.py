"""评估指标工具函数."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def calculate_adjusted_r2(r2, n, p):
    """计算调整后的 R²"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def calculate_mape(y_true, y_pred):
    """计算平均绝对百分比误差"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


def calculate_all_metrics(y_true, y_pred, n_features=1):
    """计算所有评估指标"""
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': calculate_mape(y_true, y_pred),
    }
    metrics['adj_r2'] = calculate_adjusted_r2(metrics['r2'], len(y_true), n_features)
    return metrics


def print_metrics(prefix, metrics):
    """格式化打印指标"""
    print(f"    {prefix}: R2={metrics['r2']:.4f}, Adj R2={metrics['adj_r2']:.4f}, "
          f"MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}, "
          f"RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.4f}%")
