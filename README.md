# Dynamic Patch Vision Transformer with Structured Magnitude-Based Pruning for Efficient Edge Deployment

This repo implements two core contributions, evaluated on a leaf-disease classification task (Plant Village Dataset):

1. **Adaptive Patch Size** — a Vision Transformer that doesn't use one fixed patch size. Each region of an image is scored for visual complexity with an *Information Complexity Score* (ICS = local pixel variance + Sobel gradient magnitude), then assigned a patch size of 8×8 (complex/detailed regions), 16×16 (moderate), or 32×32 (homogeneous regions) instead of a single fixed patch size like a standard ViT.
2. **Structured Magnitude-Based Pruning** — pruning whole neurons (not individual weights) in the Transformer's MLP blocks, based on the L1-norm of their incoming weight vectors, searched iteratively alongside fine-tuning to shrink the model's size/compute without letting accuracy drop past a set tolerance.

Robustness to **dynamic lighting conditions** (brightness/contrast shifts and gamma correction) is simulated and evaluated separately from accuracy under normal lighting.

## Project structure

```
adaptive_vit/                  # generic library, not tied to any specific dataset
├── config.py                  # config dataclasses (DataConfig, ICSConfig, ModelConfig, TrainConfig, PruningConfig)
├── data.py                    # dataset scanning, validation (corrupt/duplicate/near-duplicate), splitting, Dataset class
├── preprocessing.py           # transforms (resize, normalization, lighting simulation)
├── ics.py                     # Information Complexity Score & patch_size_map computation
├── patch_embedding.py         # adaptive patch extraction + DataLoader collate_fn
├── vit_model.py                # ModifiedVisionTransformer (adaptive) & BaselineVisionTransformer (fixed patch)
├── train.py                    # generic training loop
├── pruning.py                  # structured magnitude-based pruning + iterative search
├── evaluate.py                  # classification metrics, efficiency (FLOPs), baseline-vs-optimized comparison
└── visualize.py                 # all figures (training curves, pruning search, model comparison, adaptive patch map)

examples/
├── plantvillage_taxonomy.py            # dataset-specific adapter for Plant Village (species + condition -> 29 classes)
├── plantvillage_pipeline_example.py    # end-to-end reference script showing the call order (documentation, not meant to be run blindly)
└── main.ipynb                          # ready-to-run Colab notebook, cell by cell, wiring the whole pipeline above together
```

`adaptive_vit/` is fully generic: if your dataset already follows a plain `root/<class_name>/*.jpg` layout, you can skip `plantvillage_taxonomy.py` entirely and call `adaptive_vit.data.scan_imagefolder(root)` directly.

## Installation

```bash
git clone [https://github.com/Quepi14/Modified-Vision-Trransformer](https://github.com/Quepi14/Modified-Vision-Trransformer)
pip install -r requirements.txt
```

(Optional) to import `adaptive_vit` from anywhere without fiddling with `sys.path`, install it as a local package:

```bash
pip install -e .
```

## Dataset

This study uses the **Plant Village Dataset (Updated)** by `tushar5harma` on Kaggle — 9 plant species, 29 classes in total (species × condition/disease combinations).

```bash
# via the Kaggle CLI (requires kaggle.json to be set up first)
kaggle datasets download -d tushar5harma/plant-village-dataset-updated
unzip plant-village-dataset-updated.zip -d plant-village-dataset-updated
```

Once extracted, `DataConfig.dataset_root` should point to the folder that directly contains one subfolder per species.

## Usage

### Option 1 — Google Colab (easiest)

1. Upload the `adaptive_vit/` and `examples/` folders to Google Drive (or `git clone` this repo directly inside Colab).
2. Open `examples/main.ipynb` in Colab.
3. Run cell 0a (mount Drive + set up `sys.path`) and cell 0b (point it at your dataset), adjusting the paths as needed.
4. Run the rest top to bottom — each cell has a markdown heading (Cell 1 through Cell 10) following the pipeline: scan & validate the dataset → 80:10:10 split → compute mean/std → fit ICS parameters → build DataLoaders → train the Modified ViT → train the Baseline ViT (for comparison) → prune + fine-tune → evaluate on the test set → generate the figures for the results chapter.

### Option 2 — Plain Python (local machine / server)

A minimal example using the library directly (see `examples/plantvillage_pipeline_example.py` for the full version, including fitting ICS parameters and evaluation):

```python
import functools
import torch
from torch.utils.data import DataLoader

from adaptive_vit import (
    DataConfig, ICSConfig, ModelConfig, TrainConfig,
    ModifiedVisionTransformer, adaptive_collate_fn, train_model,
)
from adaptive_vit import data as avdata, preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_cfg = DataConfig(dataset_root="/path/to/plant-village-dataset-updated")
ics_cfg = ICSConfig()
train_cfg = TrainConfig()

# IMPORTANT: bind collate_fn's region_size to ics_cfg.region_size via
# functools.partial. DataLoader only calls collate_fn with a single argument
# (the batch), so adaptive_collate_fn's own default (region_size=32) does
# NOT automatically follow your ICSConfig -- if you change region_size in
# the config but forget to bind it here, patch extraction breaks silently.
collate_fn = functools.partial(adaptive_collate_fn, region_size=ics_cfg.region_size)

samples, class_names = avdata.scan_imagefolder(data_cfg.dataset_root)
valid_samples, report = avdata.validate_dataset(
    samples, class_names, min_resolution=data_cfg.image_size,
    imbalance_tolerance=data_cfg.class_imbalance_tolerance,
    check_near_duplicates=True,  # optional: also catch duplicate images, including flipped copies
)

model_cfg = ModelConfig(num_classes=len(class_names))
model = ModifiedVisionTransformer(model_cfg)
# ... build the Dataset/DataLoader objects here (see examples/plantvillage_pipeline_example.py
#     Cells 3-5 for fitting mean/std and alpha/beta/p30/p70 before this point) ...
model, history = train_model(model, train_loader, val_loader, train_cfg, device)
```

For the Plant Village dataset specifically, swap `avdata.scan_imagefolder(...)` for `plantvillage_taxonomy.scan_plantvillage_dataset(...)` — everything else stays the same.

### Full pipeline (10 stages)

1. **Scan & validate the dataset** — detect corrupt files (kept separate from low-resolution files), exact duplicates (MD5), near-duplicates/flipped copies (perceptual hash, optional), then check class balance.
2. **Stratified split** into 80:10:10 (train/val/test), per class.
3. **Compute actual mean/std** from the training subset (instead of the default ImageNet numbers) for normalization.
4. **Fit ICS parameters** (`alpha`, `beta`, `p30`, `p70`) from a sample of the validation subset — these decide the "complex vs. homogeneous" thresholds for each image region.
5. **Build the Dataset & DataLoader** — `ImageListDataset` returns `(image_tensor, patch_size_map, label)`; `adaptive_collate_fn` assembles batches out of variable-length token sequences.
6. **Train the Modified ViT** (adaptive patch size) — the main contribution.
7. **Train the Baseline ViT** (fixed 16×16 patches) — used as the comparison point in the results tables.
8. **Structured Magnitude-Based Pruning** — an iterative search over pruning ratios with fine-tuning, stopping once the accuracy drop exceeds `PruningConfig.accuracy_drop_tolerance`.
9. **Evaluate on the test set** — accuracy, efficiency (estimated FLOPs), and accuracy under simulated dynamic lighting, for both the baseline and the final pruned model.
10. **Visualization** — training curves, the pruning-ratio search plot, baseline-vs-final-model comparison, and an adaptive patch-size map for a sample image (a visual illustration of what the ICS mechanism is actually doing).

## Key configuration

Every parameter lives in `adaptive_vit/config.py` as a dataclass, so nothing is a magic number buried in the middle of the code:

| Config | Key parameters | Default |
|---|---|---|
| `DataConfig` | `image_size`, `split_ratios`, `region_size` | 224, (0.8, 0.1, 0.1), 32 |
| `ICSConfig` | `patch_size` (small/medium/large), `p_low`/`p_high` | (8, 16, 32), 30/70 (percentile) |
| `ModelConfig` | `num_classes` (**must be set explicitly**), `embed_dim`, `num_layers` | — , 384, 12 |
| `TrainConfig` | `batch_size`, `learning_rate`, `num_epochs` | 32, 3e-4, 100 |
| `PruningConfig` | `initial_prune_ratio`, `accuracy_drop_tolerance` | 0.30, 0.02 (2%) |

## Implementation notes

- Dataset validation (`data.validate_dataset`) forces `img.load()` (not just `Image.open()`) so truncated/corrupt files are caught during validation instead of crashing mid-training.
- `ImageListDataset.__getitem__` has a fallback: if a file still fails to load during training (e.g. it got corrupted after validation ran), that sample is swapped for a different random one instead of crashing the whole run.
- Near-duplicate/flip detection (`check_near_duplicates=True`) uses a self-contained perceptual hash (dHash), built with just PIL + numpy (no extra dependency), checked against both the normal orientation and a horizontal mirror — useful since Kaggle-hosted datasets are often pooled from multiple sources and can end up with the same photo (or a flipped copy of it) landing in both the train and test splits.

## Dataset citation

Plant Village Dataset (Updated), by tushar5harma, Kaggle: https://www.kaggle.com/datasets/tushar5harma/plant-village-dataset-updated
