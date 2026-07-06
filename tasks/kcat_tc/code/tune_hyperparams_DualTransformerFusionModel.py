import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from legacy_shared import DATA_DIR, DEVICE, DualTransformerFusionModel, NpyDataset, TASK_KEY, TOTAL_INPUT_DIM

k_name = TASK_KEY
TUNING_FOLD = 5
TUNING_EPOCHS = 40
TUNING_TRIALS = 50
DEFAULT_NUM_WORKERS = 4


def objective(trial):
    """
    Optuna 的目标函数，每个 trial（试验）都会执行一次这个函数。
    目标是返回验证集上的 RMSE（越小越好）。
    """

    # 1. 定义超参数的搜索空间
    # 使用 trial.suggest_ 动态获取当前试验的参数
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    seq_len = trial.suggest_categorical("seq_len", [4, 8, 16])
    dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  # 对数尺度搜索学习率
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])

    # 注意：d_model 必须能被 nhead 整除，如果不整除直接抛弃这次试验
    if d_model % nhead != 0:
        raise optuna.TrialPruned()

    # 2. 准备数据 (仅使用 Fold 1 节省调参时间)
    train_ds = NpyDataset(os.path.join(DATA_DIR, f"{k_name}_fold_{TUNING_FOLD}_train_feat.npy"),
                          os.path.join(DATA_DIR, f"{k_name}_fold_{TUNING_FOLD}_train_label.npy"))
    val_ds = NpyDataset(os.path.join(DATA_DIR, f"{k_name}_fold_{TUNING_FOLD}_val_feat.npy"),
                        os.path.join(DATA_DIR, f"{k_name}_fold_{TUNING_FOLD}_val_label.npy"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=DEFAULT_NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS)

    # 3. 初始化模型和优化器
    # 假设特征总维度是 2306
    model = DualTransformerFusionModel(
        input_dim=TOTAL_INPUT_DIM,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=seq_len,
        dropout=dropout,
        predictor_dropout_first=dropout,
        predictor_dropout_second=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 调参时 epochs 不用设置太大，快速迭代
    best_val_rmse = float('inf')

    # 4. 训练循环
    for epoch in range(TUNING_EPOCHS):
        model.train()
        for batch in train_loader:
            feat = batch['feat'].to(DEVICE)
            label = batch['label'].to(DEVICE)

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
                feat = batch['feat'].to(DEVICE)
                label = batch['label'].to(DEVICE)
                preds = model(feat)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(label.cpu().numpy())

        rmse = np.sqrt(mean_squared_error(val_labels, val_preds))

        if rmse < best_val_rmse:
            best_val_rmse = rmse

        # --------------------------------------------------------
        # Optuna 核心机制：剪枝 (Pruning)
        # 汇报当前的 RMSE，如果表现太差，Optuna 会自动终止当前 Trial
        # --------------------------------------------------------
        trial.report(rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 返回本次试验的最佳 RMSE，Optuna 会记录并尝试最小化这个值
    return best_val_rmse


if __name__ == "__main__":
    print("开始 Optuna 超参数搜索...")

    # 创建一个 Study，目标是 'minimize' (最小化 RMSE)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # 前5次试验不剪枝，每个试验前10个epoch不剪枝
    )

    # 运行 50 次试验 (Trial)，你可以根据时间情况调大 n_trials
    study.optimize(objective, n_trials=TUNING_TRIALS)

    print("\n" + "=" * 30)
    print("搜索完成！")
    print(f"最佳 RMSE: {study.best_value:.4f}")
    print("最佳参数组合:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 30)
