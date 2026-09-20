"""
examples/plantvillage_taxonomy.py

Domain-specific adapter for THIS study: the PlantVillage (Updated) dataset
(9 species, 29 leaf-condition classes total, from the Kaggle dataset by
tushar5harma). This is the only file in the whole project that knows
anything about plant species or diseases -- everything under `adaptive_vit/`
is completely generic.

If your own dataset already follows a flat `root/<class_name>/*.jpg`
ImageFolder layout, you do NOT need a file like this at all -- just call
`adaptive_vit.data.scan_imagefolder(root)` directly.

Use this file as a template for adapting `adaptive_vit` to any other
dataset whose raw layout doesn't match plain ImageFolder (e.g. nested
species/split/condition folders, or a dataset that ships with its own
train/val/test split you want to merge before re-splitting).
"""

import os
from typing import Dict, List, Tuple

from adaptive_vit.data import Sample


# Species -> list of leaf conditions, matching this study's class table.
# Folder naming convention assumed: <Species>/<split>/<Condition>/*.jpg
# Adjust this to match your actual extracted folder names if they differ
# (spacing, casing, abbreviations).
CLASS_TAXONOMY: Dict[str, List[str]] = {
    "Apple": ["Apple_Scab", "Black_Rot", "Cedar_Apple_Rust", "Healthy"],
    "Bell_Pepper": ["Bacterial_Spot", "Healthy"],
    "Cherry": ["Powdery_Mildew", "Healthy"],
    "Corn": ["Cercospora_Leaf_Spot", "Common_Rust", "Healthy", "Northern_Leaf_Blight"],
    "Grape": ["Black_Rot", "Esca_Black_Measles", "Healthy", "Leaf_Blight"],
    "Peach": ["Bacterial_Spot", "Healthy"],
    "Potato": ["Early_Blight", "Healthy", "Late_Blight"],
    "Strawberry": ["Healthy", "Leaf_Scorch"],
    "Tomato": [
        "Bacterial_Spot", "Early_Blight", "Healthy", "Late_Blight",
        "Septoria_Leaf_Spot", "Yellow_Leaf_Curl_Virus",
    ],
}

# Final list of 29 class names (order defines label index 0..28).
# Label format: "<Species>___<Condition>"
CLASS_NAMES: List[str] = [
    f"{species}___{condition}"
    for species, conditions in CLASS_TAXONOMY.items()
    for condition in conditions
]
NUM_CLASSES: int = len(CLASS_NAMES)  # -> 29


def _candidate_folder_names(name: str) -> List[str]:
    """A few naming variants a folder might use in the wild."""
    base = name.replace("_", " ")
    variants = {
        name, base, base.title(), base.lower(), base.upper(),
        name.replace("_", ""), base.replace(" ", ""),
    }
    return list(variants)


def _find_existing_dir(parent: str, name_variants: List[str]) -> str:
    if not os.path.isdir(parent):
        return ""
    lower_map = {entry.lower(): entry for entry in os.listdir(parent)}
    for variant in name_variants:
        hit = lower_map.get(variant.lower())
        if hit is not None:
            return os.path.join(parent, hit)
    return ""


def scan_plantvillage_dataset(dataset_root: str) -> Tuple[List[Sample], List[str]]:
    """Walk every species/{train,test,val}/condition folder and flatten it
    into a single Sample list, MERGING the three Kaggle-provided split
    subfolders (a fresh split is performed later via
    `adaptive_vit.data.stratified_split`).

    Returns (samples, CLASS_NAMES) -- same shape as
    `adaptive_vit.data.scan_imagefolder`, so it plugs into
    `validate_dataset` / `stratified_split` / `ImageListDataset` exactly
    the same way.
    """
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    samples: List[Sample] = []
    missing: List[str] = []

    for species, conditions in CLASS_TAXONOMY.items():
        species_dir = _find_existing_dir(dataset_root, _candidate_folder_names(species))
        if not species_dir:
            missing.append(species)
            continue

        split_dirs = []
        for split_name in ("train", "test", "val", "valid", "validation"):
            hit = _find_existing_dir(species_dir, [split_name])
            if hit:
                split_dirs.append(hit)
        if not split_dirs:
            split_dirs = [species_dir]

        for condition in conditions:
            label = class_to_idx[f"{species}___{condition}"]
            found_any = False
            for split_dir in split_dirs:
                cond_dir = _find_existing_dir(split_dir, _candidate_folder_names(condition))
                if not cond_dir:
                    continue
                for fname in os.listdir(cond_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        samples.append(Sample(filepath=os.path.join(cond_dir, fname), label=label))
                        found_any = True
            if not found_any:
                missing.append(f"{species}___{condition}")

    if missing:
        print(
            "[plantvillage_taxonomy.py] WARNING: the following folders were not "
            "found and were skipped -- check the real folder names vs. "
            "CLASS_TAXONOMY:\n  " + "\n  ".join(missing)
        )
    print(f"[plantvillage_taxonomy.py] Found {len(samples)} images "
          f"from {len(CLASS_NAMES) - len(missing)}/{len(CLASS_NAMES)} classes.")
    return samples, CLASS_NAMES