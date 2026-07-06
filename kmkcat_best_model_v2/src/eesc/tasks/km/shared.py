from __future__ import annotations

"""Shared constants and helpers for the km task branch."""

from pathlib import Path

import torch

from eesc.project import PROJECT_ROOT
from eesc.tasks.kcat.shared import (
    FEATURE_DIMENSIONS,
    CrossAttentionBlock,
    NpyDataset,
    ResidualBlock,
    SEBlock,
    TransformerBlock,
)

TASK_KEY = "km"
KM_CODE_DIR = PROJECT_ROOT / "tasks" / "km_tc" / "code"
KM_STACKING_DIR = KM_CODE_DIR / "best_model_save"
KM_DATA_DIR = KM_STACKING_DIR / "km" / "preprocessed"
DATA_DIR = "./km/preprocessed/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def legacy_data_dir() -> str:
    return DATA_DIR


def repo_data_dir() -> Path:
    return KM_DATA_DIR


def fold_file_stem(fold: int, split: str, kind: str) -> str:
    return f"fold_{fold}_{split}_{kind}.npy"


__all__ = [
    "TASK_KEY",
    "KM_CODE_DIR",
    "KM_STACKING_DIR",
    "KM_DATA_DIR",
    "DATA_DIR",
    "DEVICE",
    "FEATURE_DIMENSIONS",
    "legacy_data_dir",
    "repo_data_dir",
    "fold_file_stem",
    "NpyDataset",
    "ResidualBlock",
    "CrossAttentionBlock",
    "SEBlock",
    "TransformerBlock",
]
