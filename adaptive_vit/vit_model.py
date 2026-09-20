"""
vit_model.py

Modified Vision Transformer (Adaptive Patch Embedding + standard
Transformer Encoder), plus a `BaselineVisionTransformer` (standard ViT,
fixed 16x16 patches) you can use as a reference point for pruning and 
evaluation comparisons.

Transformer Encoder, positional-embedding mechanism, and the classification
head are shared between the baseline and the modified model (only the patch
selection & embedding stage differs) -- `TransformerBackbone` below is used
by both, so that architectural claim is enforced at the code level, not just
stated in prose.
"""

import torch
import torch.nn as nn

from .config import ModelConfig
from .patch_embedding import AdaptivePatchEmbedding

class TransformerBackbone(nn.Module):
    """
    Standard Transformer Encoder, shared by both the baseline and the
    modifieds ViT.

    Default configuration: 12 layers, 6 attention heads, embedding 
    dimension 384 -- chosen with edge deployment in mind, adjust freely
    via `ModelConfig`.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.embed_dim * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True # pre-LN, more stable when training a ViT from scratch
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.final_norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, L, D], padding_mask: [B, L] (True = ignore this token).
        Returns the CLS token's representation (position 0) after the 
        encoder:[B, D].
        """
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        encoded = self.final_norm(encoded)
        cls_representation = encoded[:, 0, :]
        return cls_representation

class ClassificationHead(nn.Module):
    """
    Standard softmax classification head (the softmax itself is applied
    implicitly by `nn.CrossEntropyLoss` during training).
    """

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_representation: torch.Tensor) -> torch.Tensor:
        return self.fc(cls_representation) # logits [B, num_classes]

class ModifiedVisionTransformer(nn.Module):
    """
    Main proposed architecture:
        Adaptive Patch Embedding -> standard Transformer Encoder -> CLS head

    Works with ANY image classification task -- just set
    `ModelConfig.num_classes` to match your dataset.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.patch_embed = AdaptivePatchEmbedding(
            embed_dim=cfg.embed_dim,
            max_position_cells=cfg.max_position_cells,
            dropout=cfg.dropout,
        )
        self.backbone = TransformerBackbone(cfg)
        self.head = ClassificationHead(cfg.embed_dim, cfg.num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        `batch`: dict produced by `patch_embedding.adaptive_collate_fn`
        (see the keys required there). Returns logits [B, num_clasess].
        """
        tokens, padding_mask = self.patch_embed(batch)
        cls_repr = self.backbone(tokens, padding_mask)
        logits = self.head(cls_repr)
        return logits

class FixedPatchEmbedding(nn.Module):
    """
    Standard ViT patch embedding: one fixed patch size for the whole
    image, one projection matrix. Used by `BaselineVisionTransformer` 
    as a reference point (not part of the adaptive-patch contribution).
    """

    def __init__(self, embed_dim: int = 384, patch_size: int = 16, image_size: int = 224, dropout: float = 0.1 ):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor):
        """
        image: [B, 3, H, W] -> tokens [B, 1+num_patches, D], paddding_mask=None
        (every image has the same number of tokens, so no masking is needer).
        """
        b = images.shape[0]
        x = self.proj(images)               # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)    # [B. num_patches, D]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.position_embedding
        x = self.dropout(x)
        return x, None

class BaselineVisionTransformer (nn.Module):
    """
    Standard ViT (Dosovitskiy et al., 2021) with fixed 16x16 patches.
    Use this as the reference "baseline" model for the pruning procedure
    and evaluation comparisons.
    """

    def __init__(self, cfg: ModelConfig, patch_size: int = 16, image_size: int = 224):
        super().__init__()
        self.patch_embed = FixedPatchEmbedding(cfg.embed_dim, patch_size, image_size, cfg.dropout)
        self.backbone = TransformerBackbone(cfg)
        self.head = ClassificationHead(cfg.embed_dim, cfg.num_classes)

    def forward(self, images: torch.Tensor) -> torch.tensor:
        tokens, padding_mask = self.patch_embed(images)
        cls_repr = self.backbone(tokens, padding_mask)
        logits = self.head(cls_repr)
        return logits