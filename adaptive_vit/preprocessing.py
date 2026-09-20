"""
preprocessing.py

Image preprocessing, augmentation, and Dynamic Lighting Simulation.

Contains:
    - `build_pre_normalize_transform` / `build_normalize_transform` -- the
        preprocessing pipeline split into two stages (see rationale below)
    - `DynamicLightingSimulation` -- simulates lighting variation
        (bright / dim / partial shadow) using 
            I' = alpha * I + beta   (brightness/contrast)
            I' = I ** gamma         (gamma correction)
    - `build_lighting_only_transform` -- lighting simulation only, used for
        the lighting-robustness evaluation scenario
    - `compute_dataset_mean_std` -- compute actual per-channel mean/std from 
        a dataset instead of hardcoding ImageNet statistics
"""

import random 
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class DynamicLightingSimulation:
    """
    Simulate lighting varation.

    Operates on an image tensor in the [0, 1] range, shape (C, H, W).
    Each call randomly picks one lighting scenario:
        - "bright"          : alpha > 1, beta > 0
        - "dim"             : alpha < 1, beta <= 0
        - "partial_shadow"  : gamma > 1 applied to aprt of the image

    alpha_range / beta_range control I' = alpha*I + beta.
    gamma_range controls I' = I ** gamma.
    """

    def __init__(
        self,
        alpha_range: Tuple[float, float] = (0.6, 1.4),
        beta_range: Tuple[float, float] = (-0.15, 0.15),
        gamma_range: Tuple[float, float] = (0.5, 2.0),
        p: float = 0.0
    ):
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img_tensor

        scenario = random.choice(["bright", "dim", "partial_shadow"])

        if scenario == "bright":
            alpha = random.uniform(1.0, self.alpha_range[1])
            beta = random.uniform(0.0, self.beta_range[1])
            out = alpha * img_tensor + beta
        elif scenario == "dim":
            alpha = random.uniform(self.alpha_range[0], 1.0)
            beta = random.uniform(self.beta_range[0], 0.0)
            out = alpha * img_tensor + beta
        else:   # partial_shadow -> gamma correction on half of the image (random side)
            gamma = random.uniform(*self.gamma_range)
            out = img_tensor.clone()
            _, h, w = out.shape
            if random.random() < 0.5:
                out[:, :, : w // 2] = out[:, :, : w // 2].clamp(min=1e-6) ** gamma
            else:
                out[:, : h // 2, :] = out[:, : h // 2, :].clamp(min=1e-6) ** gamma

        return out.clamp(0.0, 1.0)

def build_pre_normalize_transform(
    image_size: int = 224,
    split: str = "train",
    apply_lighting_simulation: bool = True,
):
    """
    FIRST half of the pipeline: resize, augmentation, ToTensor, lighting
    simulation -- BEFORE channel normalization (mean/std).

    Deliberately kept separate from `build_normalize_transform()` for two
    reasons:

    1. Local Variance Analysis and Gradient Magnitude Calculation (i.e. the 
    Information Complexity Score components) should be computed from the same
    preprocessed tensor that feeds the patch embedding step, at a point where
    pixel values are still in visually meaningful [0,1] range -- not after 
    ImageNet-style normalization, so that the measured "visual complexity" 
    isn't skewed by an otherwise arbitrary mean/std shift.

    2. Numerical-stability note: normalizing first (x-mean)/std can produce
    NEGATIVE pixel values, and gamma correction (I' = I ** gamma) raises
    a value to a fractional power -- doing that on a negative number produces
    NaN. This implementation therefore uses the order resize -> augmentation
    -> ToTensor -> lighting simulation -> (only then) normalize, so gamma 
    correction always operates on non-negative [0,1] pixel values. If you port
    this to a different pipeline ordering, watch out for this exact failure mode. 
    """
    if split == "train":
        pipeline = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(image_size, padding=8, padding_mode="reflect"),
            transforms.ToTensor(),
        ]
        if apply_lighting_simulation:
            pipeline.append(DynamicLightingSimulation())
    else:
        pipeline = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    return transforms.Compose(pipeline)

def build_normalize_transform(
    mean: Tuple[float, float, float] = (0.485, 0.465, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    """
    SECOND half of the pipeline: channel normalization only. Applied
    AFTER ICS / patch_size_map has been computed on the pre-normalization
    tensor (see `data.ImageListDataset.__getitem__`).
    """
    return transforms.Normalize(mean=mean, std=std)

def build_lighting_only_transform(image_size: int = 224):
    """
    For the lighting-robustness evaluation scenario -- Dynamic Lighting
    Simulation ONLY (no rotation/flip/crop), so the measurd accuracy dekta
    is caused purely by lighting change, not mixed with geomatric augmentation.
    `p=1.0` so the lighting effect is always applied (deterministically, unlike
    the random `p` used during training).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        DynamicLightingSimulation(p=1.0),
    ])

def build_transform(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    split: str = "train",
    apply_lighting_simulation: bool = True
):
    """
    Convience wrapper: pre-normalize + normalize combined into one Compose.
    Use this ONLY when you don't need ICS/patch_size_map (e.g. for 
    `compute_dataset_mean_std`). For actual training/evaluation, use
    `build_pre_normalize_transform` + `build_normalize_transform` separately
    through `data.ImageListDataset`.
    """
    pre = build_pre_normalize_transform(image_size, split, apply_lighting_simulation)
    return transforms.Compose([pre, build_normalize_transform(mean, std)])

def compute_dataset_mean_std(dataset, num_workers: int = 2, batch_size: int = 64):
    """
    Compute the actual per-channel RGB mean & std from a dataset (e.g. 
    the training subset), instead fo relying on default ImageNet constats.

    Cal this ONCE upfront (with a transform that has no Normalize step yet),
    then feed the result into `build_normalize_transform(mean=...,
    std=...)` for all subset (train/val/test) so they stay consistent.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_pixels = 0

    for images, _ in loader:
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])
        num_pixels += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / num_pixels
    std = (channel_sq_sum / num_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()