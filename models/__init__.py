#!/usr/bin/env python3
"""双图神经网络模型"""

from .data_types import ModelConfig, ModelOutput
from .encoder import (
    PerceiverDualEncoder,
    GNNLayer,
    ControlGatedGNNLayer,
    EDGE_TYPE_DATA,
    EDGE_TYPE_DATA_TRUE,
    EDGE_TYPE_DATA_FALSE,
    EDGE_TYPE_CONTROL,
    EDGE_TYPE_CLOCK,
    EDGE_TYPE_RESET,
    EDGE_TYPE_ENABLE,
)
from .interaction import PerceiverCrossFusion
from .hyperrectangle import HyperrectangleHead, hyperrectangle_intersection
from .contrastive_loss import compute_contrastive_loss
from .losses import compute_supervised_losses
from .model import (
    DualGraphFusionModel,
    create_model,
    create_small_model,
    create_base_model,
)

__all__ = [
    "ModelConfig",
    "ModelOutput",
    "PerceiverDualEncoder",
    "GNNLayer",
    "ControlGatedGNNLayer",
    "EDGE_TYPE_DATA",
    "EDGE_TYPE_DATA_TRUE",
    "EDGE_TYPE_DATA_FALSE",
    "EDGE_TYPE_CONTROL",
    "EDGE_TYPE_CLOCK",
    "EDGE_TYPE_RESET",
    "EDGE_TYPE_ENABLE",
    "PerceiverCrossFusion",
    "HyperrectangleHead",
    "hyperrectangle_intersection",
    "compute_contrastive_loss",
    "compute_supervised_losses",
    "DualGraphFusionModel",
    "create_model",
    "create_small_model",
    "create_base_model",
]
