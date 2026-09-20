"""
ics.py

Information Complexity Score (ICS): Local Variance Anlysis, Sobel Gradient
Magnitude Calculation, and their combination into a per-region complexity
score used to decide adaptive patch size.

All functions here operate on NUMPY arrays (not GPU tensors) because this
computation is deterministic / non-differentiable -- ICS and the patch-size 
map are computed once per image as a preprocessing step, not as something 
that gets trained.

Assumption: the image size is a multiple of `region_size` (e.g 224x224
with region_size=32 -> a 7x7 region grid). Local variance and gradient
magnitude are computed on a grayscale (single-intensity) version of the
image, consistent with the I(p) notation used in the underlying formulas.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import sobel

# Local Variance Analysis

def _to_grayscale(image_np: np.ndarray) -> np.ndarray:
    """image_np: (H, W, 3) float, range [0,1] or [0, 25]. -> (H, W) float."""
    if image_np.ndim == 2:
        return image_np.astype(np.float64)
    # Rec. 601 luma
    return (
        0.299 * image_np[..., 0] + 0.587 * image_np[..., 1] + 0.114 * image_np[...,2]
    ).astype(np.float64)

def _block_reshape(gray: np.ndarray, region_size: int) -> np.ndarray:
    """
    (H, W) -> (n_rows, n_cols, region_size*region_size), non-overlapping
    blocs. H and W must be divvisible by region_size (see module assumption).
    """
    h, w = gray.shape
    assert h % region_size == 0 and w % region_size == 0, (
        f"Image dimensions ({h}x{w}) must bne a multiple of region_size ({region_size})."
    )
    n_rows, n_cols = h // region_size, w // region_size
    blocks = gray.reshape(n_rows, region_size, n_cols, region_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(n_rows, n_cols, region_size * region_size)
    return blocks

def compute_local_variance(image_np: np.ndarray, region_size: int = 32) -> np.ndarray:
    """
    Local pixel_intensity variance per region:
        Var_local(R) = (1/K) * sum_k (p_k - mean(R))^2

    Returns: (n_rows, n_cols) array with Var_local for every region R.
    """
    gray = _to_grayscale(image_np)
    blocks = _block_reshape(gray, region_size)  # (n_rows, n_cols, K)
    mean_per_region = blocks.mean(axis=-1, keepdims=True)
    var_map = np.mean((blocks - mean_per_region) ** 2, axis=-1)
    return var_map

# Gradient Magnitude Calculation

def compute_gradient_magnitude(image_np: np.ndarray, region_size: int = 32) -> np.ndarray:
    """
    Sobel-based gradient magnitude, averaged per region:
        Gx(p) = I(p) * Sx, Gy(p) = I(p) * Sy
        ||grad I(p)|| = sqrt(Gx(p)^2) + Gy(p)^2)
        ||grad I(R)|| = (1/K) * sum_k ||grad I(p_k)||

    The Sobel filter is applied to the FULL image first (not block-by-block)
    to avoid edge artifacts at region boundaries, then averaged per region.
    Returns: (n_rows, n_cols) array with mean gradient magnitude per region R.
    """
    gray = _to_grayscale(image_np)
    gx = sobel(gray, axis=1)    # horizontal direction (columns)
    gy = sobel(gray, axis=0)    # vertical direction (rows)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    blocks = _block_reshape(grad_mag, region_size)  # (n_rows, n_cols, K)
    grad_map = blocks.mean(axis=-1)
    return grad_map

# Information Complexity Score (ICS)

def compute_ics(var_map: np.ndarray, grad_map: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """ICS(R) = alpha * Var_local(R) + beta * ||grad I(R)||"""
    return alpha * var_map + beta * grad_map

def fit_alpha_beta(
    image_arrays: List[np.ndarray], region_size: int = 32
) -> Tuple[float, float]:
    """
    Determine alpha, beta on VALIDATION SUBSET so that both components
    contribute on a comparable scale to the final score.

    Approach used here: inverse-standard-deviation scaling, i.e.
        alpha = 1 / std(Var_local over every region in the validation subset)
        beta = 1 / std(||grad I|| over every region in the validation subset) 
    so that after multiplying by alpha/beta, both components have a matching 
    scale (unit std) before being summed. this is one reasonable, explicit 
    interpretation of "fit on the validation subset for balanced contribution"
    -- document it explicitly in your metodology/report,
    since it is a design choice rather than the only valid approach.

    `image_arrays`: list of numpy arrays (H, W, 3) for the ENTIRE validation
    subset (after resizing/preprocessing, before final channel normalization).
    """
    all_var, all_grad = [], []
    for img in image_arrays:
        all_var.append(compute_local_variance(img, region_size).ravel())
        all_grad.append(compute_gradient_magnitude(img, region_size).ravel())
    all_var = np.concatenate(all_var)
    all_grad = np.concatenate(all_grad)

    std_var = all_var.std() + 1e-8
    std_grad = all_grad.std() + 1e-8
    alpha, beta = 1.0 / std_var, 1.0 /std_grad
    print(f"[ics.py] fit_alpha_beta -> alpha={alpha:.6g}, beta={beta:.6g} "
          f"(std_var={std_var:.4g}, std_grad={std_grad:.4g})")
    return alpha, beta

def fit_percentile_thresholds(
    image_arrays: List[np.ndarray],
    alpha: float,
    beta: float,
    region_size: int = 32,
    p_low: float = 30.0,
    p_high: float = 70.0,
) -> Tuple[float, float]:
    """
    Compute the low/high percentile thresholds (e.g. P30, P70) from the
    ICS distribution over EVERY region in the validation subset. Call this
    ONCE (after `fit_alpha_beta`), then store (alpha, beta, p_low, p_high)
    as fixed constans used to determine patch size for ALL images 
    (train/val/test).
    """
    all_ics = []
    for img in image_arrays:
        var_map = compute_local_variance(img, region_size)
        grad_map = compute_gradient_magnitude(img, region_size)
        all_ics.append(compute_ics(var_map, grad_map, alpha, beta).ravel())
    all_ics = np.concatenate(all_ics)

    p30 = float(np.percentile(all_ics, p_low))
    p70 = float (np.percentile(all_ics, p_high))
    print(f"[ics.py] fit_percentile_threshold -> P[p_low:g]={p30:.6g},"
          f"P[p_highLg]={p70:.6g}")
    return p30, p70

def determine_patch_size_map(
    ics_map: np.ndarray, p30: float, p70: float, patch_sizes: Tuple[int, int, int] = (8, 16, 32)
) -> np.ndarray:
    """
    Map each region's ICS value to a patch size:
        P(R) = small    if ICS(R) >= p70
        P(R) = medium   if p30 <= ICS(R) < p70
        P(R) = large    if ICS(R) < p30

    Returns: integer array (n_rows, n_cols) with values from `patch_sizes`.
    """
    small, medium, large = patch_sizes
    size_map = np.full(ics_map.shape, large, dtype=np.int32)
    size_map[(ics_map >= p30) & (ics_map < p70)] = medium
    size_map[ics_map >= p70] = small
    return size_map

def compute_patch_size_map_for_image(
    image_np: np.ndarray, alpha: float, beta: float, p30: float, p70: float,
    region_size: int = 32, patch_size: Tuple[int, int, int] = (8, 16, 32),
) -> np.ndarray:
    """
    Convenience function: raw image -> patch-size map (n_rows, n_cols).
    This is the function you call per-image during training/inference, 
    once alpha/beta/p30/p70 have been fitted from the validation subset.
    """
    var_map = compute_local_variance(image_np, region_size)
    grad_map = compute_gradient_magnitude(image_np, region_size)
    ics_map = compute_ics(var_map, grad_map, alpha, beta)
    return determine_patch_size_map(ics_map, p30, p70, patch_size)

# Patch extraction/flattening according to the size map 
# (used by patch_embedding.py)

def extract_flat_patches(
    image_np: np.ndarray, patch_size_map: np.ndarray, region_size: int = 32,
) -> Dict[int, List[Tuple[np.ndarray, int]]]:
    """
    Split an image (H, W, 3) into tokens according to `patch_size_map`
    (the output of `determine_patch_size_map`), grouped by patch size.

    Returns: dict {patch_size: [(flat_vector, position_id), ...]}
    - flat_vector : 1-D np.ndarray, the flattened patch (H_patch * W_patch * 3)
    - position_id : index of the cell on the finest 8x8 base grid
      (0 .. (H/8 * W/8 -1)), used as a shared learned positional embedding
      id in the model (see patch_embedding.py). This scheme keeps position
      meaningful/consistent even though patch size varies across images.
    """
    base_size = min(8, region_size)     # base grid follows the smalles patch size (8)
    h, w, _ = image_np.shape
    base_cols = w // base_size
    n_rows, n_cols = patch_size_map.shape

    result: Dict[int, List[Tuple[np.ndarray, int]]] = {8: [], 16: [], 32: []}

    for rr in range(n_rows):
        for rc in range(n_cols):
            p_size = int(patch_size_map[rr, rc])
            region_top = rr * region_size
            region_left = rc * region_size
            n_sub = region_size // p_size   # sub-patches per side within this region

            for i in range(n_sub):
                for j in range(n_sub):
                    top = region_top + i * p_size
                    left = region_left+ j * p_size
                    patch = image_np[top: top + p_size, left: left + p_size, :]
                    flat = patch.reshape(-1).astype(np.float32)

                    base_row = (top) // base_size
                    base_col = (left) //  base_size
                    position_id = base_row * base_cols + base_col

                    result[p_size].append((flat, position_id))

    return result