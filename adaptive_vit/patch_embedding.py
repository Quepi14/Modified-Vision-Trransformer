"""
patch_embedding.py

Adaptive Patch Embedding.

Because the number of regions assigned each patch size (8/16/32) differs
per image (it depens on that image's ICS values), a batch contains images
whose token-sequence length is not the same. The solution:

    1. `adaptive_collate_fn` (used as a DataLoader `collate_fn`) -- extracts
    and flatten each image's patches (via `ics.extract_flat_patches`), then
    PADS the token count per size (8/16/32) up to the batch maximum, producing
    rectangular tensors plus a valid-token mask.

    2. `AdaptivePatchEmbedding` (nn.Module) -- applies three separate linear 
    projections E8, E16, E32 to tokens of the matching size, adds a learned 
    positional embedding, then concatenates everything (plus a CLS token) 
    into one sequence with an attention padding mask for the Transformer Encoder.

Positional scheme: every token's positional id is the index of its cell on 
the finest 8x8 base grid (see `ics.extract_file_patches`), so tokens of size
8, 16, or 32 all amp into one shared positional-embedding table with a 
spatially consistent meaning. 
"""

from typing import Dict, List, Tuple

import numpy as np
import torch 
import torch.nn as nn

from . import ics as ics_module

PATCH_SIZES = (8, 16, 32)
PATCH_DIMS = {8: 8 * 8 * 3, 16: 16 * 16 * 3, 32: 32 * 32 * 3}

def adaptive_collate_fn(batch: List[Tuple[torch.Tensor, np.ndarray, int]], region_size: int = 32):
    """
    Custom `collate_fn` for the DataLoader.

    IMPORTANT: `region_size` here MUST match the `region_size` the dataset
    used to compute its `patch_size_map`s (`ICSConfig.region_size` /
    `DataConfig.region_size`). `DataLoader(collate_fn=...)` only accepts a
    one-argument callable, so if you use a non-default region_size, bind it
    first, e.g.:
        collate_fn=functools.partial(adaptive_collate_fn, region_size=ics_cfg.region_size)
    Leaving this at the default 32 while the dataset was built with a
    different region_size will silently corrupt patch extraction (wrong
    sub-patch counts / position ids), so don't rely on the default unless
    your config's region_size really is 32.

    `batch`: list of (normalized image_tensor [3,H,W],
                        patch_size_map [n_rows, n_cols] numpy int,
                        label int)

    Returns a dict of tensors ready for `AdaptivePatchEmbedding.forward`:
        {
            "patches_8":    FloatTensor [B, L8max, 192],
            "pos_8":        LongTensor  [B, L8max],
            "mask_8":       BoolTensor  [B, L8max]      (True = valid token),
            ... (same for 16 and 32) ...
        }
    """
    per_image_patches: List[Dict[int, List[Tuple[np.ndarray, int]]]] = []
    labels = []

    for image_tensor, patch_size_map, label in batch:
        image_np = image_tensor.permute(1, 2, 0).numpy()    # (H,W,3)
        extracted = ics_module.extract_flat_patches(image_np, patch_size_map, region_size)
        per_image_patches.append(extracted)
        labels.append(label)

    batch_size = len(batch)
    output = {"labels": torch.tensor(labels, dtype=torch.long)}

    for size in PATCH_SIZES:
        max_len = max(len(img_patches[size]) for img_patches in per_image_patches)
        max_len = max(max_len, 1)   # guard against an image with zero tokens of this size
        dim = PATCH_DIMS[size]

        patches_tensor = torch.zeros(batch_size, max_len, dim, dtype=torch.float32)
        pos_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
        mask_tensor = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for b, img_patches in enumerate(per_image_patches):
            items = img_patches[size]
            for t, (flat_vec, post_id) in enumerate(items):
                patches_tensor[b, t] = torch.from_numpy(flat_vec)
                pos_tensor[b, t] = post_id
                mask_tensor[b, t] = True

        output[f"patches_{size}"] = patches_tensor
        output[f"pos_{size}"] = pos_tensor
        output[f"mask_{size}"] = mask_tensor

    return output

class AdaptivePatchEmbedding(nn.Module):
    """
    Three separate lienar projections (E8, E16, E32) + a shared learned
    positional embedding.

    `forward` takes the dict produced by `adaptive_collate_fn` add returns:
        tokens:         FloatTensor [B, 1 + L8 + L16 + L32, embed_dim]
        padding_mask:   BoolTensor  [B, 1 + L8 + L16 + L32]
                        (True = PADDING following PyTorch's
                        `src_key_padding_mas` convention -- positions
                        marked True are IGNORED by attention)
    """ 

    def __init__(self, embed_dim: int = 384, max_position_cells: int = 784, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # E8, E16, E32 -- leanable projection matrics, one per patch size
        self.proj = nn.ModuleDict({
            str(size): nn.Linear(PATCH_DIMS[size], embed_dim) for size in PATCH_SIZES
        })

        # Shared positional embedding table (indexed via thebase 8x8 grid
        # position_id, see this module's docstring)
        self.position_embedding = nn.Embedding(max_position_cells, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for size in PATCH_SIZES:
            nn.init.trunc_normal_(self.proj[str(size)].weight, std=0.2)
            nn.init.zeros_(self.proj[str(size)].bias)
        nn.init.trunc_normal_(self.position_embedding.weight, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = batch["patches_8"].device
        batch_size = batch["patches_8"].shape[0]

        token_chunks = []
        mask_chunks = []

        for size in PATCH_SIZES:
            raw = batch[f"patches_{size}"]          # [B, L. dim]
            pos_ids = batch[f"pos_{size}"]          # [B, L]
            valid_mask = batch[f"mask_{size}"]      # [B, L] True=valid

            tok = self.proj[str(size)](raw) + self.position_embedding(pos_ids)  # [B, L, D]
            # zero out padding tokens so they can't leak into any statistics
            tok = tok * valid_mask.unsqueeze(-1)

            token_chunks.append(tok)
            mask_chunks.append(valid_mask)

        cls = self.cls_token.expand(batch_size, -1, -1)     # [B, 1, D]
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        tokens = torch.cat([cls] + token_chunks, dim=1)     # [B, 1+L8+L16+L32, D]
        valid_mask_full = torch.cat([cls_mask] + mask_chunks, dim=1)

        tokens = self.dropout(tokens)

        # PyTorch's TransformerEncoder expects src_key_padding_mask with the
        # convention True = IGNORE this token -- the inverse of our valid_mask
        padding_mask = ~valid_mask_full
        return tokens, padding_mask