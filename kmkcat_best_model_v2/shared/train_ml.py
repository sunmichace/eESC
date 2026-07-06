"""
机器学习 Stacking: 以 5 个深度学习模型的预测值作为输入, 训练 XGBoost/Ridge 元模型.

关键变化: 不再将 pH/Temperature 作为 ML 的输入特征 (已冗余).
         不同任务可配置使用不同的 ML 模型 (km/kcat 使用 XGB+Ridge, kcatkm 仅用 Ridge).
"""

import os
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge

from shared.dataset import NpyDataset
from shared.models import ALL_MODELS
from shared.metrics import calculate_all_metrics, print_metrics


def train_or_load_model(model_class, model_name, model_dir, fold, input_dim, device):
    """加载预训练的 DL 模型权重."""
    save_path = os.path.join(model_dir, f"{model_name}_fold{fold}.pth")
    model = model_class(input_dim).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    return model


def get_predictions(model, loader, device):
    """
    仅获取模型预测值和标签, 不再提取环境特征.
    """
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in loader:
            feat = batch['feat'].to(device)
            label = batch['label']
            preds = model(feat)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(label.numpy())

    return np.array(preds_list), np.array(labels_list)


def build_meta_features(preds_list):
    """
    构建元特征: [模型1_pred, 模型2_pred, ..., 模型N_pred, mean, std].
    不再包含 pH/Temp.
    """
    base_preds = np.column_stack(preds_list)
    pred_mean = np.mean(base_preds, axis=1, keepdims=True)
    pred_std = np.std(base_preds, axis=1, keepdims=True)
    meta_X = np.hstack([base_preds, pred_mean, pred_std])
    return meta_X


def run_stacking(data_dir, model_dir, device, fold_range=(1, 2), fold_prefix="",
                 batch_size=64, model_list=None, use_xgboost=True, use_ridge=True):
    """
    Stacking 主流程.

    Args:
        data_dir: preprocessed .npy 文件目录
        model_dir: DL 模型权重目录
        device: torch device
        fold_range: (start, end) fold 范围
        fold_prefix: 文件名前缀 (如 "kcatkm")
        batch_size: 批大小
        model_list: 要使用的 DL 模型列表, 默认 ALL_MODELS
        use_xgboost: 是否训练 XGBoost 元模型
        use_ridge: 是否训练 Ridge 元模型
    """
    if model_list is None:
        model_list = ALL_MODELS

    print(f"\n{'=' * 30} Start Stacking {'=' * 30}")
    print(f"ML models: XGBoost={use_xgboost}, Ridge={use_ridge}")
    print(f"Meta features: {len(model_list)} base preds + mean + std = {len(model_list) + 2} dims")
    print(f"(pH and Temperature removed from ML input)\n")

    prefix = f"{fold_prefix}_" if fold_prefix else ""

    # ========== Phase 1: 收集所有 fold 的验证集预测 ==========
    all_val_meta_features = []
    all_val_labels = []
    fold_test_data = {}

    print(">>> Phase 1: Collecting predictions from DL models...")
    for fold in range(fold_range[0], fold_range[1]):
        print(f"  Processing Fold {fold}...")

        val_ds = NpyDataset(os.path.join(data_dir, f"{prefix}fold_{fold}_val_feat.npy"),
                            os.path.join(data_dir, f"{prefix}fold_{fold}_val_label.npy"))
        test_ds = NpyDataset(os.path.join(data_dir, f"{prefix}fold_{fold}_test_feat.npy"),
                             os.path.join(data_dir, f"{prefix}fold_{fold}_test_label.npy"))

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        input_dim = val_ds[0]['feat'].shape[0]

        fold_val_preds = []
        fold_test_preds = []
        current_fold_val_labels = None
        current_fold_test_labels = None

        for ModelClass, name in model_list:
            model = train_or_load_model(ModelClass, name, model_dir, fold, input_dim, device)

            p_val, l_val = get_predictions(model, val_loader, device)
            fold_val_preds.append(p_val)
            if current_fold_val_labels is None:
                current_fold_val_labels = l_val

            p_test, l_test = get_predictions(model, test_loader, device)
            fold_test_preds.append(p_test)
            if current_fold_test_labels is None:
                current_fold_test_labels = l_test

        # 构建元特征 (不含 pH/Temp)
        X_val_meta = build_meta_features(fold_val_preds)
        all_val_meta_features.append(X_val_meta)
        all_val_labels.append(current_fold_val_labels.ravel())

        X_test_meta = build_meta_features(fold_test_preds)
        fold_test_data[fold] = (X_test_meta, current_fold_test_labels.ravel())

    # ========== Phase 2: 训练元模型 ==========
    print("\n>>> Phase 2: Training meta-models...")
    X_meta_train = np.vstack(all_val_meta_features)
    y_meta_train = np.concatenate(all_val_labels)
    print(f"    Meta training data shape: {X_meta_train.shape}")

    models = {}

    if use_xgboost:
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
            )
            xgb_model.fit(X_meta_train, y_meta_train)
            models['xgb'] = xgb_model
            print("    XGBoost trained.")
        except ImportError:
            print("    XGBoost not installed, skipping.")

    if use_ridge:
        ridge_model = Ridge(alpha=0.5)
        ridge_model.fit(X_meta_train, y_meta_train)
        models['ridge'] = ridge_model
        print("    Ridge trained.")

    # ========== Phase 3: 在测试集上评估 ==========
    print("\n>>> Phase 3: Evaluation on test sets...")
    all_results = {k: [] for k in models.keys()}
    if len(models) > 1:
        all_results['ensemble'] = []

    for fold in range(fold_range[0], fold_range[1]):
        X_test, y_test = fold_test_data[fold]
        n_features = X_test.shape[1]

        fold_preds = {}
        for name, model in models.items():
            pred = model.predict(X_test)
            fold_preds[name] = pred
            metrics = calculate_all_metrics(y_test, pred, n_features)
            all_results[name].append(metrics)
            print_metrics(f"[Fold {fold}] {name.upper()}", metrics)

        if len(models) > 1:
            # 简单平均融合
            ensemble_pred = np.mean([fold_preds[k] for k in models.keys()], axis=0)
            metrics = calculate_all_metrics(y_test, ensemble_pred, n_features)
            all_results['ensemble'].append(metrics)
            print_metrics(f"[Fold {fold}] Ensemble", metrics)

    # ========== 汇总结果 ==========
    print(f"\n{'=' * 30}")
    print("FINAL RESULTS")
    print("=" * 30)
    for name, results_list in all_results.items():
        if results_list:
            avg = {k: np.mean([m[k] for m in results_list]) for k in results_list[0]}
            print(f"  {name.upper()}:")
            print(f"    R2={avg['r2']:.4f}, Adj R2={avg['adj_r2']:.4f}, "
                  f"MAE={avg['mae']:.4f}, RMSE={avg['rmse']:.4f}, MAPE={avg['mape']:.4f}%")

    # ========== 保存模型 ==========
    save_dir = "./machine_results/"
    os.makedirs(save_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(save_dir, f"global_{name}_model.pkl")
        joblib.dump(model, path)
        print(f"  [Saved] {path}")
