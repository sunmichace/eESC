from __future__ import annotations

"""Compatibility shim from the legacy kcat code directory to the standardized package."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eesc.tasks.kcat.shared import (  # noqa: E402
    DATA_DIR,
    DEVICE,
    FEATURE_DIMENSIONS,
    TASK_KEY,
    NpyDataset,
    CrossAttentionBlock,
    ResidualBlock,
    SEBlock,
    TransformerBlock,
)
from eesc.tasks.kcat.models import (  # noqa: E402
    AttentionDTIModel,
    AttentionDTITuningModel,
    CNNPredictor,
    DualTransformerFusionModel,
    FeatureSEBlock,
    GatedInteractionModel,
    ImprovedPredictorModel,
    SimpleCrossAttentionBlock,
    STACKING_MODEL_SPECS,
    StackingAttentionDTIModel,
    StackingCNNPredictor,
    StackingDualTransformerFusionModel,
    StackingGatedInteractionModel,
    StackingImprovedPredictorModel,
    TOTAL_INPUT_DIM,
)

__all__ = [
    "DATA_DIR",
    "DEVICE",
    "FEATURE_DIMENSIONS",
    "TASK_KEY",
    "NpyDataset",
    "CrossAttentionBlock",
    "ResidualBlock",
    "SEBlock",
    "TransformerBlock",
    "TOTAL_INPUT_DIM",
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
