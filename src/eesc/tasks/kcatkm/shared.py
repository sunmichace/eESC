from __future__ import annotations

"""Shared constants and helpers for the kcatkm task branch."""

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

TASK_KEY = "kcatkm"
KCATKM_CODE_DIR = PROJECT_ROOT / "tasks" / "kcatkm_tc" / "code"
KCATKM_DATA_DIR = KCATKM_CODE_DIR / "kcatkm" / "preprocessed"
DATA_DIR = "./kcatkm/preprocessed/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def legacy_data_dir() -> str:
    return DATA_DIR


def repo_data_dir() -> Path:
    return KCATKM_DATA_DIR


def fold_file_stem(fold: int, split: str, kind: str) -> str:
    return f"{TASK_KEY}_fold_{fold}_{split}_{kind}.npy"


__all__ = [
    "TASK_KEY",
    "KCATKM_CODE_DIR",
    "KCATKM_DATA_DIR",
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
