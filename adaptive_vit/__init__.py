"""
adaptive_vit

A reusable Vision Transformer variant whose patch size adapts per-region to 
local visual complexity (Information Complexity Score = local pixel 
variance + Sobel gradient magnitude), combined with structured magnitude-base
pruning for edge deployment.

Quick import:

    from adaptive_vit import (
        ModifiedVisionTransformer, BaselineVisionTransformer,
        ModelConfig, TrainConfig, PruningConfig, DataConfig, ICSConfig,
        train_model, iterative_prune_and_finetune,
    )
"""

from .config import DataConfig, ICSConfig, ModelConfig, TrainConfig, PruningConfig
from .vit_model import ModifiedVisionTransformer, BaselineVisionTransformer, TransformerBackbone
from .patch_embedding import AdaptivePatchEmbedding, adaptive_collate_fn
from .train import train_model, run_epoch
from .pruning import iterative_prune_and_finetune, structured_magnitude_pruning
from . import ics
from . import data
from . import preprocessing
from . import evaluate
from . import visualize

__all__ = [
    "DataConfig", "ICSConfig", "ModelConfig", "TrainConfig", "PruningConfig",
    "ModifiedVisionTransformer", "BaselineVisionTransformer", "TransformerBackbone",
    "AdaptivePatchEmbedding", "adaptive_collate_fn", 
    "train_model", "run_epoch",
    "iterative_prune_and_finetune", "structured_magnitude_pruning",
    "ics", "data", "preprocessing", "evaluate", "visualize" 
]