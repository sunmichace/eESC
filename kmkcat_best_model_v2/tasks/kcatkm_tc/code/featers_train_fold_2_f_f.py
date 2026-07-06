import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from legacy_shared import (
    AttentionDTIModel,
    DATA_DIR,
    DEVICE,
    DualTransformerFusionModel,
    GatedInteractionModel,
    ImprovedPredictorModel,
    NpyDataset,
    TASK_KEY,
    TOTAL_INPUT_DIM,
)

k_name = TASK_KEY
ACTIVE_TRAIN_FOLDS = range(1, 2)
DEFAULT_NUM_WORKERS = 4
DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 64


def run_5fold_training_improved(epochs=100, batch_size=512):
    print(f"\n{'=' * 10} Start Improved 5-Fold Training {'=' * 10}")

    sample_feat = np.load(os.path.join(DATA_DIR, f"{k_name}_fold_1_train_feat.npy"))
    raw_input_dim = sample_feat.shape[1]
    print(f"Detected raw feature dimension: {raw_input_dim}")

    fold_results = []

    for fold in ACTIVE_TRAIN_FOLDS:
        print(f"\n--- Processing Fold {fold}/5 ---")

        train_ds = NpyDataset(
            os.path.join(DATA_DIR, f"{k_name}_fold_{fold}_train_feat.npy"),
            os.path.join(DATA_DIR, f"{k_name}_fold_{fold}_train_label.npy"),
        )
        val_ds = NpyDataset(
            os.path.join(DATA_DIR, f"{k_name}_fold_{fold}_val_feat.npy"),
            os.path.join(DATA_DIR, f"{k_name}_fold_{fold}_val_label.npy"),
        )
        test_ds = NpyDataset(
            os.path.join(DATA_DIR, f"{k_name}_fold_{fold}_test_feat.npy"),
            os.path.join(DATA_DIR, f"{k_name}_fold_{fold}_test_label.npy"),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=DEFAULT_NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS)

        model = AttentionDTIModel(input_dim=raw_input_dim or TOTAL_INPUT_DIM).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        criterion = nn.MSELoss()
        best_val_rmse = float("inf")
        patience_counter = 0
        patience = 15

        for epoch in range(epochs):
            model.train()
            loop = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch + 1}/{epochs}", leave=False)
            for batch in loop:
                feat = batch["feat"].to(DEVICE)
                label = batch["label"].to(DEVICE)

                optimizer.zero_grad()
                preds = model(feat)
                loss = criterion(preds, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    feat = batch["feat"].to(DEVICE)
                    label = batch["label"].to(DEVICE)
                    preds = model(feat)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(label.cpu().numpy())

            rmse = np.sqrt(mean_squared_error(val_labels, val_preds))
            r2 = r2_score(val_labels, val_preds)
            scheduler.step(rmse)

            if rmse < best_val_rmse:
                best_val_rmse = rmse
                patience_counter = 0
                torch.save(model.state_dict(), f"improved_best_model_fold{fold}.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  -> Early stopping at epoch {epoch + 1}")
                    break

            print(f"  -> Val RMSE: {rmse:.4f} | R2: {r2:.4f} | Best RMSE: {best_val_rmse:.4f}")

        model.load_state_dict(torch.load(f"improved_best_model_fold{fold}.pth"))
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                feat = batch["feat"].to(DEVICE)
                preds = model(feat)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(batch["label"].cpu().numpy())

        test_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
        test_r2 = r2_score(test_labels, test_preds)
        print(f"  [Fold {fold} Result] RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")
        fold_results.append({"rmse": test_rmse, "r2": test_r2})

    print("\n" + "=" * 30)
    print("Final Improved Results")
    print("=" * 30)
    print(f"Avg RMSE: {np.mean([result['rmse'] for result in fold_results]):.4f}")
    print(f"Avg R2: {np.mean([result['r2'] for result in fold_results]):.4f}")


if __name__ == "__main__":
    run_5fold_training_improved(epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE)
