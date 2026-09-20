"""
examples/plantvillage_pipeline_example.py

End-to-end example wiring the generic `adaptive_vit` library together with
the PlantVillage-specific adapter (`plantvillage_taxonomy.py`) for this
study. Copy/adapt the cells below into a Google Colab notebook (each
"# --- Cell N ---" block is roughly one notebook cell).

Not meant to run unattended start-to-finish -- it's a call-order reference /
checklist, not a guaranteed one-shot script.

For a dataset that already follows a plain `root/<class_name>/*.jpg` layout,
skip `plantvillage_taxonomy` entirely and call
`adaptive_vit.data.scan_imagefolder(root)` directly instead of
`scan_plantvillage_dataset(root)` in Cell 1 -- everything else stays the
same.
"""

import functools
import sys
import os

import torch
from torch.utils.data import DataLoader

# Make sure both the adaptive_vit package and this examples/ folder are on
# the path (adjust if your folder layout in Colab differs):
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adaptive_vit import (
    DataConfig, ICSConfig, ModelConfig, TrainConfig, PruningConfig,
    ModifiedVisionTransformer, BaselineVisionTransformer,
    adaptive_collate_fn, train_model,
)
from adaptive_vit import data as avdata
from adaptive_vit import preprocessing, ics, evaluate, pruning

from plantvillage_taxonomy import scan_plantvillage_dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = DataConfig(dataset_root="/content/plant-village-dataset-updated")
    ics_cfg = ICSConfig()
    train_cfg = TrainConfig()
    prune_cfg = PruningConfig()

    # adaptive_collate_fn's `region_size` must match the region_size the
    # dataset used to build its patch_size_maps (ics_cfg.region_size) --
    # DataLoader only calls collate_fn with one argument (the batch), so we
    # bind region_size here instead of relying on adaptive_collate_fn's
    # own default (which is only correct if ics_cfg.region_size == 32).
    collate_fn = functools.partial(adaptive_collate_fn, region_size=ics_cfg.region_size)

    # --- Cell 1: scan & validate the dataset ---
    # (swap this line for `avdata.scan_imagefolder(data_cfg.dataset_root)`
    #  if you switch to a plain ImageFolder-style dataset)
    samples, class_names = scan_plantvillage_dataset(data_cfg.dataset_root)
    valid_samples, report = avdata.validate_dataset(
        samples, class_names, min_resolution=data_cfg.image_size,
        imbalance_tolerance=data_cfg.class_imbalance_tolerance,
    )
    print("Classes exceeding the imbalance tolerance:", report["classes_exceeding_tolerance"])

    model_cfg = ModelConfig(num_classes=len(class_names))  # -> 29 for this study

    # --- Cell 2: stratified 80:10:10 split ---
    train_samples, val_samples, test_samples = avdata.stratified_split(
        valid_samples, ratios=data_cfg.split_ratios, seed=data_cfg.split_seed,
    )

    # --- Cell 3: compute actual mean/std from the TRAIN subset ---
    from PIL import Image
    import numpy as np

    pre_tf_novaug = preprocessing.build_pre_normalize_transform(
        data_cfg.image_size, split="test", apply_lighting_simulation=False)

    class _RawImageDataset(torch.utils.data.Dataset):
        def __init__(self, samples_):
            self.samples = samples_
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            img = Image.open(self.samples[idx].filepath).convert("RGB")
            return pre_tf_novaug(img), self.samples[idx].label

    mean, std = preprocessing.compute_dataset_mean_std(_RawImageDataset(train_samples))
    print("Training-subset mean/std:", mean, std)

    # --- Cell 4: fit alpha, beta, p30, p70 from a VALIDATION SUBSAMPLE ---
    sample_val = val_samples[:400]
    val_images_np = [
        pre_tf_novaug(Image.open(s.filepath).convert("RGB")).permute(1, 2, 0).numpy()
        for s in sample_val
    ]
    alpha, beta = ics.fit_alpha_beta(val_images_np, region_size=ics_cfg.region_size)
    p30, p70 = ics.fit_percentile_thresholds(
        val_images_np, alpha, beta, region_size=ics_cfg.region_size,
        p_low=ics_cfg.p_low, p_high=ics_cfg.p_high,
    )

    # --- Cell 5: build the final train/val/test Dataset & DataLoader objects ---
    def make_dataset(samples_, split):
        return avdata.ImageListDataset(
            samples_,
            pre_normalize_transform=preprocessing.build_pre_normalize_transform(
                data_cfg.image_size, split=split, apply_lighting_simulation=(split == "train")),
            normalize_transform=preprocessing.build_normalize_transform(mean, std),
            ics_alpha=alpha, ics_beta=beta, ics_p30=p30, ics_p70=p70,
            region_size=ics_cfg.region_size,
        )

    train_ds = make_dataset(train_samples, "train")
    val_ds = make_dataset(val_samples, "val")
    test_ds = make_dataset(test_samples, "test")

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    # --- Cell 6: build & train the Modified Vision Transformer ---
    modified_model = ModifiedVisionTransformer(model_cfg)
    modified_model, history = train_model(modified_model, train_loader, val_loader, train_cfg, device)

    pre_prune_metrics = evaluate.compute_classification_metrics(modified_model, val_loader, device)
    print("Modified ViT accuracy (before pruning) on val:", pre_prune_metrics.accuracy)

    # --- Cell 7 (optional, for the comparison table): train the Baseline ViT ---
    baseline_model = BaselineVisionTransformer(model_cfg)
    baseline_train_loader = DataLoader(avdata.PlainImageDataset(train_ds), batch_size=train_cfg.batch_size,
                                        shuffle=True, num_workers=4)
    baseline_val_loader = DataLoader(avdata.PlainImageDataset(val_ds), batch_size=train_cfg.batch_size,
                                      shuffle=False, num_workers=2)
    baseline_model, _ = train_model(baseline_model, baseline_train_loader, baseline_val_loader, train_cfg, device)

    # --- Cell 8: Structured Magnitude-Based Pruning + fine-tuning ---
    prune_result = pruning.iterative_prune_and_finetune(
        modified_model,
        baseline_accuracy=pre_prune_metrics.accuracy,
        train_loader=train_loader, val_loader=val_loader,
        cfg=prune_cfg, device=device,
    )
    final_model = prune_result.model
    print(f"Final prune ratio: {prune_result.final_ratio:.2%}, "
          f"val_acc: {prune_result.val_accuracy:.4f}")

    # --- Cell 9: evaluate on the TEST SET (accuracy, efficiency, lighting robustness) ---
    lighting_test_ds = avdata.ImageListDataset(
        test_samples,
        pre_normalize_transform=preprocessing.build_lighting_only_transform(data_cfg.image_size),
        normalize_transform=preprocessing.build_normalize_transform(mean, std),
        ics_alpha=alpha, ics_beta=beta, ics_p30=p30, ics_p70=p70,
        region_size=ics_cfg.region_size,
    )
    lighting_loader = DataLoader(lighting_test_ds, batch_size=train_cfg.batch_size, shuffle=False,
                                  collate_fn=collate_fn, num_workers=2)
    baseline_lighting_loader = DataLoader(avdata.PlainImageDataset(lighting_test_ds), batch_size=train_cfg.batch_size,
                                           shuffle=False, num_workers=2)
    baseline_test_loader = DataLoader(avdata.PlainImageDataset(test_ds), batch_size=train_cfg.batch_size,
                                       shuffle=False, num_workers=2)

    from adaptive_vit import visualize

    comparison_report = evaluate.compare_baseline_vs_optimized(
        baseline_model=baseline_model,
        optimized_model=final_model,
        baseline_test_loader=baseline_test_loader,
        optimized_test_loader=test_loader,
        baseline_lighting_loader=baseline_lighting_loader,
        optimized_lighting_loader=lighting_loader,
        device=device,
    )

    # --- Cell 10: figures for BAB IV / results section ---
    visualize.plot_training_curves(history, save_path="fig_training_curves.png")
    visualize.plot_pruning_search(prune_result.history, save_path="fig_pruning_search.png")
    visualize.plot_model_comparison(comparison_report, save_path="fig_model_comparison.png")

    # Adaptive patch-size map for one sample test image, to illustrate the
    # core ICS-driven mechanism. NOTE: we deliberately re-derive the PRE-
    # normalization [0,1] image here (via pre_tf_novaug), NOT
    # `test_ds[0]`'s returned tensor -- that one has already gone through
    # `normalize_transform` (ImageNet-style mean/std), which can produce
    # negative values that would just get clipped to black/white and ruin
    # the figure. The patch_size_map itself is recomputed the same way
    # `ImageListDataset.__getitem__` does internally, so it matches exactly
    # what the model actually saw for this image.
    sample = test_samples[0]
    sample_raw = Image.open(sample.filepath).convert("RGB")
    sample_pre_tensor = pre_tf_novaug(sample_raw)
    sample_image_np = sample_pre_tensor.permute(1, 2, 0).numpy()
    sample_patch_map = ics.compute_patch_size_map_for_image(
        sample_image_np, alpha, beta, p30, p70, region_size=ics_cfg.region_size,
    )
    visualize.visualize_patch_size_map(sample_image_np, sample_patch_map,
                                        region_size=ics_cfg.region_size,
                                        save_path="fig_patch_size_map.png")


if __name__ == "__main__":
    main()