"""5 种深度学习模型定义 + 辅助模块 (Blocks)."""

import torch
import torch.nn as nn


# ===================== 辅助模块 (Blocks) =====================

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        q = query.unsqueeze(1)
        k = key.unsqueeze(1)
        v = value.unsqueeze(1)
        attn_out, _ = self.multihead_attn(q, k, v)
        attn_out = attn_out.squeeze(1)
        x = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = x.view(b, c, 1)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y)
        return x * y


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=1024, dropout=0.1):
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

    def forward(self, x, query_source):
        attn_output, _ = self.self_attn(query=query_source, key=x, value=x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x


# ===================== 5 种深度学习模型 =====================
# 特征布局: [smiles(1024) | prot(1280) | env(2)] = 2306
SMI_DIM = 1024
PROT_DIM = 1280
ENV_DIM = 2


# ---------- 模型 1: ImprovedPredictorModel ----------
class ImprovedPredictorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, interaction_dim=512):
        super().__init__()
        self.smi_dim = SMI_DIM
        self.prot_dim = PROT_DIM
        self.env_dim = ENV_DIM
        self.interaction_dim = interaction_dim

        self.smi_proj = nn.Linear(self.smi_dim, self.interaction_dim)
        self.prot_proj = nn.Linear(self.prot_dim, self.interaction_dim)

        mlp_input_dim = self.interaction_dim * 3 + self.env_dim
        self.head = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        smi_feat = x[:, :self.smi_dim]
        prot_feat = x[:, self.smi_dim: self.smi_dim + self.prot_dim]
        env_feat = x[:, self.smi_dim + self.prot_dim:]

        smi_p = self.smi_proj(smi_feat)
        prot_p = self.prot_proj(prot_feat)
        interaction_feat = smi_p * prot_p

        combined = torch.cat([smi_p, prot_p, interaction_feat, env_feat], dim=1)
        return self.head(combined)


# ---------- 模型 2: GatedInteractionModel ----------
class GatedInteractionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.smi_dim = SMI_DIM
        self.prot_dim = PROT_DIM
        self.env_dim = ENV_DIM

        self.smi_encoder = nn.Sequential(
            nn.Linear(self.smi_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.prot_encoder = nn.Sequential(
            nn.Linear(self.prot_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.env_gate = nn.Sequential(
            nn.Linear(self.env_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        smi = x[:, :self.smi_dim]
        prot = x[:, self.smi_dim: self.smi_dim + self.prot_dim]
        env = x[:, self.smi_dim + self.prot_dim:]

        h_smi = self.smi_encoder(smi)
        h_prot = self.prot_encoder(prot)
        g_env = self.env_gate(env)

        interaction = h_smi * h_prot
        interaction = interaction * g_env

        combined = torch.cat([interaction, (h_smi + h_prot) / 2], dim=1)
        return self.head(combined)


# ---------- 模型 3: AttentionDTIModel ----------
class AttentionDTIModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.smi_dim = SMI_DIM
        self.prot_dim = PROT_DIM
        self.env_dim = ENV_DIM

        self.smi_proj = nn.Sequential(
            nn.Linear(self.smi_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(self.prot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.cross_attn_d2p = CrossAttentionBlock(hidden_dim, num_heads=8, dropout=dropout)
        self.cross_attn_p2d = CrossAttentionBlock(hidden_dim, num_heads=8, dropout=dropout)

        self.env_net = nn.Sequential(
            nn.Linear(self.env_dim, 64),
            nn.GELU()
        )

        fusion_dim = hidden_dim * 4 + 64
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            SEBlock(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        smi = x[:, :self.smi_dim]
        prot = x[:, self.smi_dim: self.smi_dim + self.prot_dim]
        env = x[:, self.smi_dim + self.prot_dim:]

        h_smi = self.smi_proj(smi)
        h_prot = self.prot_proj(prot)
        h_env = self.env_net(env)

        feat_d2p = self.cross_attn_d2p(h_smi, h_prot, h_prot)
        feat_p2d = self.cross_attn_p2d(h_prot, h_smi, h_smi)

        combined = torch.cat([feat_d2p, feat_p2d, h_smi, h_prot, h_env], dim=1)
        feat = self.fusion_net(combined)
        return self.head(feat)


# ---------- 模型 4: DualTransformerFusionModel ----------
class DualTransformerFusionModel(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=4, num_layers=2, seq_len=8):
        super().__init__()
        self.smi_raw_dim = SMI_DIM
        self.prot_raw_dim = PROT_DIM
        self.env_dim = ENV_DIM
        self.seq_len = seq_len
        self.d_model = d_model

        self.smi_proj = nn.Sequential(
            nn.Linear(self.smi_raw_dim, seq_len * d_model),
            nn.LayerNorm(seq_len * d_model),
            nn.GELU()
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(self.prot_raw_dim, seq_len * d_model),
            nn.LayerNorm(seq_len * d_model),
            nn.GELU()
        )

        self.left_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead=nhead, dropout=0.2) for _ in range(num_layers)
        ])
        self.right_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead=nhead, dropout=0.2) for _ in range(num_layers)
        ])

        concat_dim = (seq_len * d_model) * 2 + self.env_dim
        self.predictor = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        smi_feat = x[:, :self.smi_raw_dim]
        prot_feat = x[:, self.smi_raw_dim: self.smi_raw_dim + self.prot_raw_dim]
        env_feat = x[:, self.smi_raw_dim + self.prot_raw_dim:]
        batch_size = x.size(0)

        feat_L = self.smi_proj(smi_feat).view(batch_size, self.seq_len, self.d_model)
        feat_KB = self.prot_proj(prot_feat).view(batch_size, self.seq_len, self.d_model)

        curr_L = feat_L
        curr_KB = feat_KB
        for i in range(len(self.left_layers)):
            prev_L = curr_L
            prev_KB = curr_KB
            curr_L = self.left_layers[i](x=prev_L, query_source=prev_KB)
            curr_KB = self.right_layers[i](x=prev_KB, query_source=prev_L)

        flat_L = curr_L.flatten(start_dim=1)
        flat_KB = curr_KB.flatten(start_dim=1)
        combined = torch.cat([flat_L, flat_KB, env_feat], dim=1)
        return self.predictor(combined)


# ---------- 模型 5: CNNPredictor ----------
class CNNPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.smi_dim = SMI_DIM
        self.prot_dim = PROT_DIM
        self.env_dim = ENV_DIM

        self.smi_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=8, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(8),
        )
        self.prot_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=8, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(8),
        )

        flat_dim = 32 * 8 * 2 + self.env_dim
        self.head = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        smi = x[:, :self.smi_dim].unsqueeze(1)
        prot = x[:, self.smi_dim: self.smi_dim + self.prot_dim].unsqueeze(1)
        env = x[:, self.smi_dim + self.prot_dim:]

        smi_out = self.smi_conv(smi).flatten(start_dim=1)
        prot_out = self.prot_conv(prot).flatten(start_dim=1)

        combined = torch.cat([smi_out, prot_out, env], dim=1)
        return self.head(combined)


# ===================== 模型注册表 =====================
ALL_MODELS = [
    (ImprovedPredictorModel, "ImprovedPredictorModel"),
    (GatedInteractionModel, "GatedInteractionModel"),
    (AttentionDTIModel, "AttentionDTIModel"),
    (DualTransformerFusionModel, "DualTransformerFusionModel"),
    (CNNPredictor, "CNNPredictor"),
]
