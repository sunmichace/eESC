from __future__ import annotations

"""Reusable km models built on the same feature layout as kcat."""

from eesc.tasks.kcat.models import (
    AttentionDTIModel,
    AttentionDTITuningModel,
    CNNPredictor,
    DualTransformerFusionModel,
    ENVIRONMENT_DIM,
    FeatureSEBlock,
    GatedInteractionModel,
    ImprovedPredictorModel,
    KcatModelBase,
    PROTEIN_DIM,
    SMILES_DIM,
    SimpleCrossAttentionBlock,
    TOTAL_INPUT_DIM,
)

KmModelBase = KcatModelBase


class StackingImprovedPredictorModel(ImprovedPredictorModel):
    """Checkpoint-compatible improved predictor used by legacy km stacking."""

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        hidden_dim: int = 1024,
        interaction_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            interaction_dim=interaction_dim,
            dropout=dropout,
        )


class StackingGatedInteractionModel(GatedInteractionModel):
    """Checkpoint-compatible gated interaction model used by legacy km stacking."""

    def __init__(self, input_dim: int = TOTAL_INPUT_DIM, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)


class StackingAttentionDTIModel(AttentionDTIModel):
    """Checkpoint-compatible attention model used by legacy km stacking."""

    def __init__(self, input_dim: int = TOTAL_INPUT_DIM, hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=8,
            env_hidden_dim=64,
        )


class StackingDualTransformerFusionModel(DualTransformerFusionModel):
    """Checkpoint-compatible transformer model used by legacy km stacking."""

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        seq_len: int = 16,
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
    """Checkpoint-compatible CNN baseline used by legacy km stacking."""

    def __init__(self, input_dim: int = TOTAL_INPUT_DIM, hidden_dim: int = 1024, dropout: float = 0.4):
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
    "KmModelBase",
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
