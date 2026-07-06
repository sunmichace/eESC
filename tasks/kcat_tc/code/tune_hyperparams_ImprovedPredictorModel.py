import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from legacy_shared import DATA_DIR, DEVICE, ImprovedPredictorModel, NpyDataset, TASK_KEY, TOTAL_INPUT_DIM

k_name = TASK_KEY
TUNING_FOLD = 5
TUNING_EPOCHS = 40
TUNING_TRIALS = 40
DEFAULT_NUM_WORKERS = 4


def objective(trial):
    # 1. 定义 MLP 专属的超参数搜索空间
    # 交互维度：决定了药物和靶点在多大的特征空间内进行相乘
    interaction_dim = trial.suggest_categorical("interaction_dim", [128, 256, 512])
    # 隐藏层维度：决定了预测头的容量
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    # 正则化参数
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

    # 优化器参数
    lr = trial.suggest_float("lr", 1e-5, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # 2. 准备数据 (仅使用 Fold 1 节省时间)
    train_ds = NpyDataset(os.path.join(DATA_DIR, f"{k_name}_fold_{TUNING_FOLD}_train_feat.npy"),
                          os.path.join(DATA_DIR, f"{k_name}_fold_{TUNING_FOLD}_train_label.npy"))
    val_ds = NpyDataset(os.path.join(DATA_DIR, f"{k_name}_fold_{TUNING_FOLD}_val_feat.npy"),
                        os.path.join(DATA_DIR, f"{k_name}_fold_{TUNING_FOLD}_val_label.npy"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=DEFAULT_NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS)

    # 3. 初始化模型和优化器
    model = ImprovedPredictorModel(
        input_dim=TOTAL_INPUT_DIM,
        hidden_dim=hidden_dim,
        interaction_dim=interaction_dim,
        dropout=dropout
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # MLP 收敛通常比 Transformer 快且稳，设置 40-50 个 epoch 足够看出趋势
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

        # Optuna 剪枝机制
        trial.report(rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_rmse


if __name__ == "__main__":
    print(f"{'=' * 10} Start Optuna Tuning for ImprovedPredictorModel {'=' * 10}")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # 因为参数组合没 Transformer 那么苛刻，跑 40 次 trial 应该就能找到很好的解了
    study.optimize(objective, n_trials=TUNING_TRIALS)

    print("\n" + "=" * 30)
    print("搜索完成！")
    print(f"最佳 RMSE: {study.best_value:.4f}")
    print("最佳参数组合:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 30)
