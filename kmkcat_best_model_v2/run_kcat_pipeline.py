#!/usr/bin/env python3
"""
kcat 任务完整流程: 特征提取 → DL 训练 → ML Stacking.
用法: python run_kcat_pipeline.py [--extract] [--train-dl] [--train-ml]
"""

import sys
import torch
sys.path.insert(0, ".")

from configs import kcat_config as cfg
from shared.extract_features import run_extraction
from shared.train_dl import run_all_models
from shared.train_ml import run_stacking


def get_device():
    return torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    do_extract = "--extract" in sys.argv or len(sys.argv) == 1
    do_train_dl = "--train-dl" in sys.argv or len(sys.argv) == 1
    do_train_ml = "--train-ml" in sys.argv or len(sys.argv) == 1
    device = get_device()

    if do_extract:
        print("\n========== Step 1: Feature Extraction ==========")
        run_extraction(
            target_col=cfg.TARGET_COL,
            data_dir=cfg.DATA_DIR,
            save_dir=cfg.PREPROCESSED_DIR,
            device=device,
            fold_range=cfg.FOLD_RANGE,
        )

    if do_train_dl:
        print("\n========== Step 2: Deep Learning Training ==========")
        run_all_models(
            data_dir=cfg.PREPROCESSED_DIR,
            device=device,
            epochs=cfg.DL_EPOCHS,
            batch_size=cfg.DL_BATCH_SIZE,
            fold_range=cfg.DL_FOLD_RANGE,
            fold_prefix=cfg.FOLD_PREFIX,
            save_dir=".",
        )

    if do_train_ml:
        print("\n========== Step 3: ML Stacking ==========")
        run_stacking(
            data_dir=cfg.PREPROCESSED_DIR,
            model_dir=".",
            device=device,
            fold_range=cfg.ML_FOLD_RANGE,
            fold_prefix=cfg.FOLD_PREFIX,
            batch_size=cfg.ML_BATCH_SIZE,
            use_xgboost=cfg.ML_USE_XGBOOST,
            use_ridge=cfg.ML_USE_RIDGE,
        )
