#!/usr/bin/env python3
"""双图神经网络模型"""

from .data_types import (
    MODEL_ARCHITECTURE_PERCEIVER_FUSION,
    MODEL_ARCHITECTURE_POOLED_ADD,
    MODEL_ARCHITECTURE_RTL_GCN,
    MODEL_ARCHITECTURES,
    ModelConfig,
    ModelOutput,
)
from .encoder import (
    DualGraphEncoder,
    PerceiverDualEncoder,
    GNNLayer,
    ControlGatedGNNLayer,
    RTLGCNLayer,
    EDGE_TYPE_DATA,
    EDGE_TYPE_DATA_TRUE,
    EDGE_TYPE_DATA_FALSE,
    EDGE_TYPE_CONTROL,
    EDGE_TYPE_CLOCK,
    EDGE_TYPE_RESET,
    EDGE_TYPE_ENABLE,
)
from .interaction import PerceiverCrossFusion
from .hyperrectangle import (
    HyperrectangleHead,
    hyperrectangle_geometry,
)
from .contrastive_loss import compute_coverage_geometry_losses
from .losses import compute_supervised_losses
from .model import (
    DualGraphFusionModel,
    PooledAddBaselineModel,
    GCNFusionBaselineModel,
    create_model,
    create_small_model,
    create_base_model,
)

__all__ = [
    "ModelConfig",
    "ModelOutput",
    "MODEL_ARCHITECTURE_PERCEIVER_FUSION",
    "MODEL_ARCHITECTURE_POOLED_ADD",
    "MODEL_ARCHITECTURE_RTL_GCN",
    "MODEL_ARCHITECTURES",
    "DualGraphEncoder",
    "PerceiverDualEncoder",
    "GNNLayer",
    "ControlGatedGNNLayer",
    "RTLGCNLayer",
    "EDGE_TYPE_DATA",
    "EDGE_TYPE_DATA_TRUE",
    "EDGE_TYPE_DATA_FALSE",
    "EDGE_TYPE_CONTROL",
    "EDGE_TYPE_CLOCK",
    "EDGE_TYPE_RESET",
    "EDGE_TYPE_ENABLE",
    "PerceiverCrossFusion",
    "HyperrectangleHead",
    "hyperrectangle_geometry",
    "compute_coverage_geometry_losses",
    "compute_supervised_losses",
    "DualGraphFusionModel",
    "PooledAddBaselineModel",
    "GCNFusionBaselineModel",
    "create_model",
    "create_small_model",
    "create_base_model",
]
