from __future__ import annotations

"""Reusable kcat model definitions extracted from the legacy experiment scripts."""

import torch
import torch.nn as nn

from eesc.tasks.kcat.shared import FEATURE_DIMENSIONS, CrossAttentionBlock, SEBlock, TransformerBlock

SMILES_DIM = FEATURE_DIMENSIONS["smiles"]
PROTEIN_DIM = FEATURE_DIMENSIONS["protein"]
ENVIRONMENT_DIM = FEATURE_DIMENSIONS["environment"]
TOTAL_INPUT_DIM = FEATURE_DIMENSIONS["total"]


class KcatModelBase(nn.Module):
    """Base class that encapsulates the legacy feature layout."""

    def __init__(self, input_dim: int = TOTAL_INPUT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.smi_dim = SMILES_DIM
        self.prot_dim = PROTEIN_DIM
        self.env_dim = ENVIRONMENT_DIM

    def split_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        smi = x[:, : self.smi_dim]
        prot = x[:, self.smi_dim : self.smi_dim + self.prot_dim]
        env = x[:, self.smi_dim + self.prot_dim :]
        return smi, prot, env


class CNNPredictor(KcatModelBase):
    """1D-CNN baseline that encodes SMILES and protein embeddings separately."""

    def __init__(self, input_dim: int = TOTAL_INPUT_DIM, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__(input_dim=input_dim)

        self.smi_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )

        self.prot_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )

        cnn_out_dim = (64 * 16) * 2 + self.env_dim
        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smi, prot, env = self.split_features(x)
        smi_out = self.smi_cnn(smi.unsqueeze(1)).flatten(1)
        prot_out = self.prot_cnn(prot.unsqueeze(1)).flatten(1)
        combined = torch.cat([smi_out, prot_out, env], dim=1)
        return self.head(combined)


class ImprovedPredictorModel(KcatModelBase):
    """Project both modalities into a shared latent space and model their interaction."""

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        hidden_dim: int = 512,
        interaction_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__(input_dim=input_dim)
        self.interaction_dim = interaction_dim

        self.smi_proj = nn.Linear(self.smi_dim, self.interaction_dim)
        self.prot_proj = nn.Linear(self.prot_dim, self.interaction_dim)

        mlp_input_dim = self.interaction_dim * 3 + self.env_dim
        self.head = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smi_feat, prot_feat, env_feat = self.split_features(x)
        smi_p = self.smi_proj(smi_feat)
        prot_p = self.prot_proj(prot_feat)
        interaction_feat = smi_p * prot_p
        combined = torch.cat([smi_p, prot_p, interaction_feat, env_feat], dim=1)
        return self.head(combined)


class GatedInteractionModel(KcatModelBase):
    """Environment-aware gated interaction model."""

    def __init__(self, input_dim: int = TOTAL_INPUT_DIM, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__(input_dim=input_dim)

        self.smi_encoder = nn.Sequential(
            nn.Linear(self.smi_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.prot_encoder = nn.Sequential(
            nn.Linear(self.prot_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.env_gate = nn.Sequential(
            nn.Linear(self.env_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smi, prot, env = self.split_features(x)
        h_smi = self.smi_encoder(smi)
        h_prot = self.prot_encoder(prot)
        g_env = self.env_gate(env)
        interaction = (h_smi * h_prot) * g_env
        combined = torch.cat([interaction, (h_smi + h_prot) / 2], dim=1)
        return self.head(combined)


class DualTransformerFusionModel(KcatModelBase):
    """Bidirectional transformer fusion model for SMILES and protein features."""

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = 8,
        dropout: float = 0.2,
        predictor_dropout_first: float | None = None,
        predictor_dropout_second: float | None = None,
    ):
        super().__init__(input_dim=input_dim)
        self.seq_len = seq_len
        self.d_model = d_model
        predictor_dropout_first = 0.3 if predictor_dropout_first is None else predictor_dropout_first
        predictor_dropout_second = dropout if predictor_dropout_second is None else predictor_dropout_second

        self.smi_proj = nn.Sequential(
            nn.Linear(self.smi_dim, seq_len * d_model),
            nn.LayerNorm(seq_len * d_model),
            nn.GELU(),
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(self.prot_dim, seq_len * d_model),
            nn.LayerNorm(seq_len * d_model),
            nn.GELU(),
        )

        self.left_layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead=nhead, dropout=dropout) for _ in range(num_layers)]
        )
        self.right_layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead=nhead, dropout=dropout) for _ in range(num_layers)]
        )

        concat_dim = (seq_len * d_model) * 2 + self.env_dim
        self.predictor = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(predictor_dropout_first),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(predictor_dropout_second),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smi_feat, prot_feat, env_feat = self.split_features(x)
        batch_size = x.size(0)

        feat_l = self.smi_proj(smi_feat).view(batch_size, self.seq_len, self.d_model)
        feat_kb = self.prot_proj(prot_feat).view(batch_size, self.seq_len, self.d_model)

        curr_l = feat_l
        curr_kb = feat_kb
        for left_layer, right_layer in zip(self.left_layers, self.right_layers):
            prev_l = curr_l
            prev_kb = curr_kb
            curr_l = left_layer(x=prev_l, query_source=prev_kb)
            curr_kb = right_layer(x=prev_kb, query_source=prev_l)

        flat_l = curr_l.flatten(start_dim=1)
        flat_kb = curr_kb.flatten(start_dim=1)
        combined = torch.cat([flat_l, flat_kb, env_feat], dim=1)
        return self.predictor(combined)


class AttentionDTIModel(KcatModelBase):
    """Cross-attention fusion model used by the main legacy 5-fold trainer."""

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        num_heads: int = 8,
        env_hidden_dim: int = 64,
    ):
        super().__init__(input_dim=input_dim)

        self.smi_proj = nn.Sequential(
            nn.Linear(self.smi_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.prot_proj = nn.Sequential(
            nn.Linear(self.prot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cross_attn_d2p = CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_p2d = CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)

        self.env_net = nn.Sequential(
            nn.Linear(self.env_dim, env_hidden_dim),
            nn.GELU(),
        )

        fusion_dim = hidden_dim * 4 + env_hidden_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            SEBlock(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smi, prot, env = self.split_features(x)
        h_smi = self.smi_proj(smi)
        h_prot = self.prot_proj(prot)
        h_env = self.env_net(env)

        feat_d2p = self.cross_attn_d2p(h_smi, h_prot, h_prot)
        feat_p2d = self.cross_attn_p2d(h_prot, h_smi, h_smi)

        combined = torch.cat([feat_d2p, feat_p2d, h_smi, h_prot, h_env], dim=1)
        feat = self.fusion_net(combined)
        return self.head(feat)


class FeatureSEBlock(nn.Module):
    """Feature-wise SE variant used by the legacy attention tuning script."""

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class SimpleCrossAttentionBlock(nn.Module):
    """Lightweight cross-attention block preserved from the Optuna tuning variant."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        query_seq = query.unsqueeze(1)
        key_seq = key.unsqueeze(1)
        value_seq = value.unsqueeze(1)
        attn_out, _ = self.attention(query_seq, key_seq, value_seq)
        attn_out = attn_out.squeeze(1)
        return self.norm(query + attn_out)


class AttentionDTITuningModel(KcatModelBase):
    """Legacy Optuna attention variant kept as a separate shared implementation."""

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_heads: int = 8,
        env_hidden_dim: int = 64,
    ):
        super().__init__(input_dim=input_dim)

        self.smi_proj = nn.Sequential(
            nn.Linear(self.smi_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.prot_proj = nn.Sequential(
            nn.Linear(self.prot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cross_attn_d2p = SimpleCrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_p2d = SimpleCrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)

        self.env_net = nn.Sequential(
            nn.Linear(self.env_dim, env_hidden_dim),
            nn.GELU(),
        )

        fusion_dim = hidden_dim * 4 + env_hidden_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            FeatureSEBlock(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smi, prot, env = self.split_features(x)
        h_smi = self.smi_proj(smi)
        h_prot = self.prot_proj(prot)
        h_env = self.env_net(env)

        feat_d2p = self.cross_attn_d2p(h_smi, h_prot, h_prot)
        feat_p2d = self.cross_attn_p2d(h_prot, h_smi, h_smi)

        combined = torch.cat([feat_d2p, feat_p2d, h_smi, h_prot, h_env], dim=1)
        feat = self.fusion_net(combined)
        return self.head(feat)


class StackingImprovedPredictorModel(ImprovedPredictorModel):
    """Checkpoint-compatible improved predictor used by the legacy stacking scripts."""

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        hidden_dim: int = 1024,
        interaction_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            interaction_dim=interaction_dim,
            dropout=dropout,
        )


class StackingGatedInteractionModel(GatedInteractionModel):
    """Checkpoint-compatible gated interaction model used by the legacy stacking scripts."""

    def __init__(self, input_dim: int = TOTAL_INPUT_DIM, hidden_dim: int = 1024, dropout: float = 0.2):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)


class StackingAttentionDTIModel(AttentionDTIModel):
    """Checkpoint-compatible attention model used by the legacy stacking scripts."""

    def __init__(self, input_dim: int = TOTAL_INPUT_DIM, hidden_dim: int = 1024, dropout: float = 0.2):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=8,
            env_hidden_dim=64,
        )


class StackingDualTransformerFusionModel(DualTransformerFusionModel):
    """Checkpoint-compatible transformer fusion model used by the legacy stacking scripts."""

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 1,
        seq_len: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            seq_len=seq_len,
            dropout=dropout,
            predictor_dropout_first=dropout,
            predictor_dropout_second=dropout,
        )


class StackingCNNPredictor(CNNPredictor):
    """Checkpoint-compatible CNN baseline used by the legacy stacking scripts."""

    def __init__(self, input_dim: int = TOTAL_INPUT_DIM, hidden_dim: int = 512, dropout: float = 0.6):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)


STACKING_MODEL_SPECS = (
    (StackingImprovedPredictorModel, "ImprovedPredictorModel"),
    (StackingGatedInteractionModel, "GatedInteractionModel"),
    (StackingAttentionDTIModel, "AttentionDTIModel"),
    (StackingDualTransformerFusionModel, "DualTransformerFusionModel"),
    (StackingCNNPredictor, "CNNPredictor"),
)


__all__ = [
    "SMILES_DIM",
    "PROTEIN_DIM",
    "ENVIRONMENT_DIM",
    "TOTAL_INPUT_DIM",
    "KcatModelBase",
    "CNNPredictor",
    "ImprovedPredictorModel",
    "GatedInteractionModel",
    "DualTransformerFusionModel",
    "AttentionDTIModel",
    "FeatureSEBlock",
    "SimpleCrossAttentionBlock",
    "AttentionDTITuningModel",
    "StackingImprovedPredictorModel",
    "StackingGatedInteractionModel",
    "StackingAttentionDTIModel",
    "StackingDualTransformerFusionModel",
    "StackingCNNPredictor",
    "STACKING_MODEL_SPECS",
]
