"""
深度学习 5 折交叉验证训练: 对每种模型训练 5 个 fold, 保存最佳权重.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from shared.dataset import NpyDataset
from shared.models import ALL_MODELS


def train_single_model(model_class, model_name, data_dir, device, fold=1,
                       epochs=300, batch_size=64, lr=5e-4, weight_decay=1e-2, patience=15,
                       fold_prefix="", save_dir="."):
    """
    对单个模型在单个 fold 上训练.

    Returns:
        (test_rmse, test_r2)
    """
    prefix = f"{fold_prefix}_" if fold_prefix else ""
    train_ds = NpyDataset(os.path.join(data_dir, f"{prefix}fold_{fold}_train_feat.npy"),
                          os.path.join(data_dir, f"{prefix}fold_{fold}_train_label.npy"))
    val_ds = NpyDataset(os.path.join(data_dir, f"{prefix}fold_{fold}_val_feat.npy"),
                        os.path.join(data_dir, f"{prefix}fold_{fold}_val_label.npy"))
    test_ds = NpyDataset(os.path.join(data_dir, f"{prefix}fold_{fold}_test_feat.npy"),
                         os.path.join(data_dir, f"{prefix}fold_{fold}_test_label.npy"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    raw_input_dim = train_ds[0]['feat'].shape[0]
    model = model_class(input_dim=raw_input_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    patience_counter = 0
    save_path = os.path.join(save_dir, f"{model_name}_fold{fold}.pth")

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{epochs}", leave=False)
        for batch in loop:
            feat = batch['feat'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            preds = model(feat)
            loss = criterion(preds, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                feat = batch['feat'].to(device)
                label = batch['label'].to(device)
                val_preds.extend(model(feat).cpu().numpy())
                val_labels.extend(label.cpu().numpy())

        rmse = np.sqrt(mean_squared_error(val_labels, val_preds))
        r2 = r2_score(val_labels, val_preds)
        scheduler.step(rmse)

        if rmse < best_val_rmse:
            best_val_rmse = rmse
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break

    # 测试
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            feat = batch['feat'].to(device)
            test_preds.extend(model(feat).cpu().numpy())
            test_labels.extend(batch['label'].cpu().numpy())

    test_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
    test_r2 = r2_score(test_labels, test_preds)
    print(f"  [Fold {fold} Result] RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")
    return test_rmse, test_r2


def run_5fold_training(model_class, model_name, data_dir, device, epochs=300, batch_size=64,
                       fold_range=(1, 2), fold_prefix="", save_dir="."):
    """
    对单个模型跑 5 折 (或指定范围) 训练.
    """
    print(f"\n{'='*10} Start {model_name} Training {'='*10}")

    fold_results = []
    for fold in range(fold_range[0], fold_range[1]):
        print(f"\n--- Processing Fold {fold}/{fold_range[1]-1} ---")
        rmse, r2 = train_single_model(
            model_class, model_name, data_dir, device,
            fold=fold, epochs=epochs, batch_size=batch_size,
            fold_prefix=fold_prefix, save_dir=save_dir
        )
        fold_results.append({'rmse': rmse, 'r2': r2})

    avg_rmse = np.mean([r['rmse'] for r in fold_results])
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    print(f"\n{'='*30}")
    print(f"Final {model_name} Results: Avg RMSE={avg_rmse:.4f}, Avg R2={avg_r2:.4f}")
    return avg_rmse, avg_r2


def run_all_models(data_dir, device, epochs=300, batch_size=64,
                   fold_range=(1, 2), fold_prefix="", save_dir=".",
                   model_list=None):
    """
    对所有注册的模型依次训练.
    """
    if model_list is None:
        model_list = ALL_MODELS

    all_results = {}
    for model_class, name in model_list:
        avg_rmse, avg_r2 = run_5fold_training(
            model_class, name, data_dir, device,
            epochs=epochs, batch_size=batch_size,
            fold_range=fold_range, fold_prefix=fold_prefix, save_dir=save_dir
        )
        all_results[name] = {'rmse': avg_rmse, 'r2': avg_r2}

    print(f"\n{'='*30}")
    print("All Models Summary")
    for name, res in all_results.items():
        print(f"  {name}: RMSE={res['rmse']:.4f}, R2={res['r2']:.4f}")
    return all_results
