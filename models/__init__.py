#!/usr/bin/env python3
"""双图神经网络模型"""

from .data_types import ModelConfig, ModelOutput
from .encoder import InteractiveDualEncoder, GNNLayer
from .interaction import CrossGraphInteraction, CrossGraphAttention
from .model import (
    DualGraphFusionModel,
    create_model,
    create_small_model,
    create_base_model,
)

__all__ = [
    "ModelConfig",
    "ModelOutput",
    "InteractiveDualEncoder",
    "GNNLayer",
    "CrossGraphInteraction",
    "CrossGraphAttention",
    "DualGraphFusionModel",
    "create_model",
    "create_small_model",
    "create_base_model",
]
