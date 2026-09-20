"""
data.py

Generic dataset scanning, validation, splitting, and PyTorch `Dataset`
wrpper with the ICS / adaptive-patch-size computation integrated into its
`__getitem__`.

this module makes NO assumption about yout specific class taxonomy. The
main entry point, `scan_imagefolder`, expects the same layout as
`torchvision.datasets.ImageFOlder`:

    <root>/<class_name_a>/*.jpg
    <root>/<class_name_b>/*.jpg
    ...

Class name are inferred automatically (sorted alphabetically, same 
convention as torchvision), so this works out of the box for any image
classification dataset arraged this way -- not just the leaf-disease
study this library was originally built for.

If your raw dataset has a different layout (e.g. nested species/split/condition
folder, or a split that ships with train/val/test subfolders your want to merge
before re-splitting), write a small adapter that returns a `List[Sample]` in the
same shape `scan_imagefolder` produces, then feed it into `validate_dataset` /
`stratified_split` / `ImageListDataset` exactly the same way.

See `example/plantvillage_taxonomy.py` for a worked example of such an adapter.
"""

import hashlib
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

@dataclass
class Sample:
    filepath: str
    label: int      #index into whatever class_name list you used

#Generic ImageFolder-style scanning

def scan_imagefolder(
    dataset_root: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> Tuple[List[Sample], List[str]]:
    """Scan standard `<root>/<class_name>/*.ext` folder layout.

    Returns (samples, class_name). `class_name[label]` give the
    human-readable name for a given integer label -- keep this list
    around, you will need it for `ModelConfig(num_classes=len(class_name))`
    and for turning prediction s back into readbale class names.
    """
    class_names = sorted(
        d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples: List[Sample] = []
    for name in class_names:
        class_dir = os.path.join(dataset_root, name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(extensions):
                samples.append(Sample(filepath=os.path.join(class_dir, fname), label=class_to_idx[name]))

    print(f"[data.py] Found {len(samples)} image across {len(class_names)} classes in {dataset_root}")
    return samples, class_names


#Validation (duplicates, near-duplicates/flips, corruption, resolution, class-imbalance report)
def _file_hash(filepath: str, block_size: int = 65536) -> str:
    """MD5 of the raw file bytes -- only catches EXACT, byte-for-byte
    duplicate files. A flipped, rotated, resized, or re-compressed copy of
    the same underlying photo produces a completely different hash and
    will NOT be caught by this function (see `_perceptual_hash_bits` /
    `check_near_duplicates` below for that case)."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
        return hasher.hexdigest()


def _perceptual_hash_bits(gray_image: Image.Image, hash_size: int = 8) -> np.ndarray:
    """
    Difference-hash (dHash) of an already-opened, already-grayscale PIL
    image: shrink to (hash_size+1) x hash_size, then for each row encode
    "is this pixel brighter than the one to its left?" as one bit.

    Unlike `_file_hash` (MD5 of raw bytes), two images that look almost
    identical to a human -- the same photo saved at a different quality,
    resized, or (when also hashing the horizontally-mirrored version, see
    `validate_dataset`) flipped left-right -- produce hashes that differ in
    only a handful of bits, so they can be caught via Hamming distance
    instead of needing to match exactly.
    """
    small = gray_image.resize((hash_size + 1, hash_size), Image.BILINEAR)
    pixels = np.asarray(small, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).ravel()

def validate_dataset(
        samples: List[Sample],
        class_names: List[str],
        min_resolution: int = 224,
        imbalance_tolerance: float = 0.15,
        check_duplicates: bool = True,
        check_near_duplicates: bool = False,
        near_duplicate_hash_size: int = 8,
        near_duplicate_max_distance: int = 5,
) -> Tuple[List[Sample], Dict]:
    """Apply validation criteria:
    1. Corrupt / unreadable files (a full pixel decode is attempted, not
        just reading the file header -- see note below)
    2. Duplicate images (MD5 hash of file content -- exact, byte-for-byte
        duplicates only)
    3. (optional) Near-duplicate images, INCLUDING horizontally-flipped
        copies of the same underlying photo (perceptual hash -- see
        `check_near_duplicates` below)
    4. Minimum resolution (`min_resolution` x `min_resolution`)
    5. Label consistency (implicit -- every scanned image already carries a label)
    6. Per-class image-count distribution (flag class imbalance beyond
        `imbalance_tolerance` of the mean count per class)

    Returns (valid_samples, report_dict). Images failing the corrupt,
    duplicate, near-duplicate, or resolution check are REMOVED. Class
    imbalance is only REPORTED (e.g. for later use as class weight during
    training), not used to drop data.

    Corruption check note: `Image.open()` alone only reads the file's
    header (PIL opens lazily), so a truncated/partially-corrupt file (e.g.
    a download that got cut off) can still "open" successfully and only
    fail later, mid-training, when something finally forces a full decode.
    This function calls `img.load()` to force that decode NOW, during
    validation, so such files are caught and reported as `dropped_corrupt`
    up front instead of crashing a training run hours in.

    `check_near_duplicates`: in addition to the exact-hash check above,
    also flags near-duplicates using a perceptual hash (dHash) computed on
    BOTH the image's original orientation and its horizontal mirror -- this
    catches copies of the same underlying photo that were flipped, lightly
    re-compressed, or resized before being re-uploaded into the dataset,
    none of which `_file_hash` (MD5 of raw bytes) can see at all. This
    matters most when the raw data was pooled from several sources (as
    with the Kaggle-hosted PlantVillage mirrors): an original photo and an
    unlabeled flipped copy of it landing in different splits is a real
    train/test leakage risk, since the model would then be evaluated on
    content it has effectively already seen. Off by default because it is
    O(n^2) in the number of images kept so far (each new image is compared
    against every previously-kept one) and noticeably slower than the exact
    check -- fine for datasets up to a few tens of thousands of images; for
    much larger datasets, consider validating a random subsample first.
    `near_duplicate_max_distance`: two images are flagged as near-duplicates
    when their dHash Hamming distance is <= this many bits out of
    `near_duplicate_hash_size ** 2` total bits (default 5 / 64 bits, i.e.
    roughly 92% of the hash bits must agree).
    """
    seen_hashes = set()
    seen_phash_bits: List[np.ndarray] = []  # only populated if check_near_duplicates
    valid: List[Sample] = []
    dropped_corrupt = 0
    dropped_duplicate = 0
    dropped_near_duplicate = 0
    dropped_resolution = 0

    for s in samples:
        try:
            with Image.open(s.filepath) as img:
                img.load()  # force full pixel decode -- see corruption note above
                w, h = img.size
                if check_near_duplicates:
                    gray = img.convert("L")
                    normal_bits = _perceptual_hash_bits(gray, near_duplicate_hash_size)
                    flipped_bits = _perceptual_hash_bits(
                        ImageOps.mirror(gray), near_duplicate_hash_size)
        except Exception:
            dropped_corrupt += 1
            continue

        if w < min_resolution or h < min_resolution:
            dropped_resolution += 1
            continue

        if check_duplicates:
            file_hash = _file_hash(s.filepath)
            if file_hash in seen_hashes:
                dropped_duplicate += 1
                continue
            seen_hashes.add(file_hash)

        if check_near_duplicates:
            is_near_duplicate = False
            if seen_phash_bits:
                stacked = np.stack(seen_phash_bits)               # [n_seen, n_bits]
                dist_normal = (stacked != normal_bits).sum(axis=1)
                dist_flipped = (stacked != flipped_bits).sum(axis=1)
                closest = min(dist_normal.min(), dist_flipped.min())
                is_near_duplicate = closest <= near_duplicate_max_distance
            if is_near_duplicate:
                dropped_near_duplicate += 1
                continue
            seen_phash_bits.append(normal_bits)

        valid.append(s)

    counts = np.zeros(len(class_names), dtype=int)
    for s in valid:
        counts[s.label] += 1
    mean_count = counts[counts > 0].mean() if (counts > 0).any() else 0
    imbalanced_classes = [
        class_names[i]
        for i,c in enumerate(counts)
        if c > 0 and abs(c - mean_count) / max(mean_count, 1) > imbalance_tolerance
    ]

    report = {
        "total_input": len(samples),
        "total_valid": len(valid),
        "dropped_corrupt": dropped_corrupt,
        "dropped_duplicate": dropped_duplicate,
        "dropped_near_duplicate": dropped_near_duplicate,
        "dropped_resolution": dropped_resolution,
        "class_counts": {class_names[i]: int(c) for i, c in enumerate(counts)},
        "mean_count_per_class": float(mean_count),
        "imbalance_tolerance": imbalance_tolerance,
        "classes_exceeding_tolerance": imbalanced_classes,
    }
    print(
        f"[data.py] Validation done: {report['total_valid']}/{report['total_input']} "
        f"images passed. Corrupt/unreadable dropped: {dropped_corrupt}, "
        f"exact duplicates dropped: {dropped_duplicate}, "
        + (f"near-duplicates (incl. flips) dropped: {dropped_near_duplicate}, "
           if check_near_duplicates else "")
        + f"low resolution dropped: {dropped_resolution}. "
        f"Classes exceeding imbalance tolerance ({imbalance_tolerance: .0%}): "
        f"{len(imbalanced_classes)} classes."
    )

    return valid, report


#Splitting
def stratified_split(
    samples: List[Sample],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Stratified train:val:test split, proportional per class.

    Run this once and save the resulting filepath lists (e.g. to a .json
    file), so every later stage refers to exactly the same subsets.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6,   "Split ratios must sum to 1.0"
    train_ratio, val_ratio, test_ratio = ratios
    labels = [s.label for s in samples]

    train_samples, temp_samples, _train_labels, temp_labels = train_test_split(
        samples, labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        train_size=relative_val,
        stratify=temp_labels,
        random_state=seed,
    ) 

    print(
        f"[data.py] Split done -> train: {len(train_samples)}, "
        f"val: {len(val_samples)}, test: {len(test_samples)}"
    )
    return train_samples, val_samples, test_samples

#torch.utils.data.Dataset wrapper with ICS Integrated
class ImageListDataset(Dataset):
    """Pytorch Dataset wrapper around a List[Sample], with adaptive
    patch-size computation build into `__getitem__`
    
    Each call follow this pipeline:
        raw image
            -> pre_normalize_transform (resize, augmentation, ToTensor,
                optional lighting/other photometric simulation)
            -> compute_patch_size_map (ICS: Local Variance + Sobel Gradien
                Magnitude)
            -> normalize_tranform (channel mean/std)
            -> (nomalized_image_tensor, patch_size_map, label)
    
    `ics_alpha, ics_beta, ics_p30, ics_p70` MUST come from
    `ice.fit_alpha_beta` + `ics.fit_percentile_thresholds`, run ONCE
    upfront on your validation subset, then reused as fixed constants 
    for the train/val/test subsets.

    if you only want a standart fixed-patch ViT (`BaselineVisionTransformer`
    in `vit_model.py`), you don't need this class at all - any plain Dataset
    returning (image_tensor, label) works with the default collate.
    """

    def __init__(
        self, 
        samples: List[Sample],
        pre_normalize_transform,
        normalize_transform,
        ics_alpha: float,
        ics_beta: float,
        ics_p30: float,
        ics_p70: float,
        region_size: int = 32,
    ):
        self.samples = samples
        self.pre_normalize_transform = pre_normalize_transform
        self.normalize_transform = normalize_transform
        self.ics_alpha = ics_alpha
        self.ics_beta = ics_beta
        self.ics_p30 = ics_p30
        self.ics_p70 = ics_p70
        self.region_size = region_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from . import ics as ics_module #local import keeps this class standalone

        sample = self.samples[idx]
        try:
            image = Image.open(sample.filepath).convert("RGB")
        except Exception as e:
            # Defensive fallback only: `validate_dataset(..., )` should
            # already have removed every unreadable file BEFORE this class
            # is ever built, so this is expected to be rare. But if a file
            # somehow still fails here (e.g. it got corrupted on disk AFTER
            # validation ran, or validation was skipped), don't let one bad
            # file crash an entire training run hours in -- fall back to a
            # different, randomly-chosen sample instead of raising.
            print(f"[data.py] WARNING: failed to load '{sample.filepath}' ({e}); "
                  f"substituting a different sample instead.")
            return self.__getitem__(random.randrange(len(self.samples)))

        pre_tensor = self.pre_normalize_transform(image)    # (3,H,W) in [0.1]
        pre_np = pre_tensor.permute(1, 2, 0).numpy()        # (H,W,#)

        patch_size_map = ics_module.compute_patch_size_map_for_image(
            pre_np, self.ics_alpha, self.ics_beta, self.ics_p30, self.ics_p70,
            region_size=self.region_size,
        )

        image_tensor = self.normalize_transform(pre_tensor)

        return image_tensor, patch_size_map, sample.label

class PlainImageDataset(Dataset):
    """Thin wrapper that strips the patch_size_map of an 
    `ImageListDataset`, returning plain (image, label) pairs -- convient
    when you want to reuse the same underlying samples/transform to also
    train a `BaselineVisionTransformer` with the standard PyTorch collate.
    """
    def __init__(self, base: ImageListDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, _patch_size_map, label = self.base[idx]
        return image, label