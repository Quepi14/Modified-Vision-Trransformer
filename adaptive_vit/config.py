"""
config.py

Generic configuration dataclasses for the adaptive_vit library. Nothing in
this fileis tied to any specific dataset -- `ModelConfig.num_classes` is a 
required argument you supply based on YOUR dataset (e.g. `len(class_names)`
from `data.scan_imagefolder`), not a hardcoded constant.
"""

from dataclasses import dataclass

@dataclass
class DataConfig:
    """Dataset loading, validation, and preprocessing settings."""
    dataset_root: str
    image_size: int = 224                           # Minimum resolution & resize target
    class_imbalance_tolerance: float = 0.15 
    split_ratios: tuple = (0.8, 0.1, 0.1)           # Train : Val : Test
    split_seed: int = 42
    region_size: int = 32                           # Grid region size used for ICS
    normalize_mean: tuple = (0.485, 0.456, 0.406)   # Placeholder; ideally recomputed
    normalize_std: tuple = (0.229, 0.224, 0.225)    # From your own training subset

@dataclass
class ICSConfig:
    """Information Complexity Score (ICS) settings."""
    region_size: int = 32
    patch_size: tuple = (8, 16, 32) # Small -> Complex region, large -> homogeneous region
    p_low: float = 30.0             # Set lower percentile threshold
    p_high: float = 70.0            # Set upper percentile threshold
    # Alpha, beta should be ideally be fitted by ics.fit_alpha_beta() on your
    # validation subset (see that function's docstring); there are only a
    # fallback default.
    alpha: float = 1.0
    beta: float = 1.0

@dataclass
class ModelConfig:
    """Adaptive Patch Embedding + Modified Vision Transformer settings.
    
    `num_classes` has no default on purpose -- always pass it explicitly
    (e.g. `ModelConfig(num_classes=len(class_names))`) so it is never 
    silently wrong for your dataset.
    """
    num_classes: int
    embed_dim: int = 384
    num_layers: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    in_channels: int = 3
    # Positional-embedding table length = number of cells in the fines 8x8
    # grid over a 224x244 image -> (224 / 8) ** 2 = 784. See the position-id
    # scheme explained in patch_embedding.py, Recompute this if you change 
    # `image_size` away from 224.
    max_position_cells: int = (224 // 8) ** 2

@dataclass
class TrainConfig:
    """Training loop settings."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    num_epochs: int = 100
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0

@dataclass
class PruningConfig:
    """Structured magnitude-based pruning settings."""
    initial_prune_ratio: float = 0.30 
    ratio_step: float = 0.05    # Telling how much ratio is reduced per failed iteration
    accuracy_drop_tolerance: float = 0.02  #2%
    finetune_epochs: int = 10
    finetune_lr: float = 3e-5   # Lower than the initial training LR
    max_prune_iterations: int = 5