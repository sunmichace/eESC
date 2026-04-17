from __future__ import annotations

"""Shared constants, data helpers, and reusable blocks for the kcat task."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from eesc.project import PROJECT_ROOT

TASK_KEY = "kcat"
KCAT_CODE_DIR = PROJECT_ROOT / "tasks" / "kcat_tc" / "code"
KCAT_DATA_DIR = KCAT_CODE_DIR / "kcat" / "preprocessed"
DATA_DIR = "./kcat/preprocessed/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIMENSIONS = {
    "smiles": 1024,
    "protein": 1280,
    "environment": 2,
    "total": 2306,
}


def legacy_data_dir() -> str:
    """Return the legacy relative data directory expected by historical scripts."""
    return DATA_DIR


def repo_data_dir() -> Path:
    """Return the repository-absolute preprocessed data directory."""
    return KCAT_DATA_DIR


def fold_file_stem(fold: int, split: str, kind: str) -> str:
    return f"{TASK_KEY}_fold_{fold}_{split}_{kind}.npy"


class NpyDataset(Dataset):
    """Minimal dataset wrapper used across the legacy kcat training scripts."""

    def __init__(self, feat_path: str | Path, label_path: str | Path):
        self.features = np.load(feat_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "feat": torch.tensor(self.features[idx]),
            "label": torch.tensor(self.labels[idx]),
        }


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = query.unsqueeze(1)
        k = key.unsqueeze(1)
        v = value.unsqueeze(1)

        attn_out, _ = self.multihead_attn(q, k, v)
        attn_out = attn_out.squeeze(1)

        x = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for simple feature recalibration."""

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels = x.size()
        y = x.view(batch_size, channels, 1)
        y = self.avg_pool(y).view(batch_size, channels)
        y = self.fc(y)
        return x * y


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, query_source: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(query=query_source, key=x, value=x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        return self.norm2(x)

