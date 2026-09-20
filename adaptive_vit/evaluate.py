"""
evaluate.py

Model evaluation -- three scenarios:

    1. Classification accuracy      ->  `compute_classification_metrics`
    2. Computational efficiency     ->  `count_parameters`, `estimate_transformer_flops`,
                                        `measure_inference_time`
    3. Lighting robustness          ->  `lighting_robustness_test`
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from .train import _forward_and_labels

# Scenario 1 - Classification accuracy
@dataclass 
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float

def compute_classification_metrics(model: nn.Module, loader, device: torch.device) ->  ClassificationMetrics:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits, labels, = _forward_and_labels(model, batch, device)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    return ClassificationMetrics(
        accuracy=acc,
        precision_macro=p_macro, recall_macro=r_macro, f1_macro=f1_macro,
        precision_weighted=p_weighted, recall_weighted=r_weighted, f1_weighted=f1_weighted
    )


# Scenario 2- Computional efficiency
def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return {"total": total, "trainable": trainable, "nonzero": nonzero}

def estimate_transformer_flops(
    seq_len: int, embed_dim: int, num_layers: int, num_heads: int, mlp_ratio: float = 4.0
) -> int :
    """
    ANALYTICAL FLOPs estimate for one forward pass of a standard
    Transformer Encoder (not the reuslt of a profiler -- meant as a 
    relative proxy for comparing the Baseline vs. the Final Optimized
    model, NOT an exact, profiler-grade numer). Per-layer formula
    (following the common approximation used in Transformer scaling-law
    literature, e.g. Kaplan et al. 2020):

        FLOPs_attention_proj    = 4 * seq_len * embed_dim^2     (Q,K,V,O projections)
        FLOPs_attention_score   = 2 * seq_len^2 * embed_dim     (QK^T and softmax*V)
        FLOPs_mlp               = 2 * (2 * seq_len * embed_dim * (embed_dim*mlp_ratio))

    Several terms are multiplied by 2 because FLOPs = 2 x MACs.
    """
    hidden = int(embed_dim * mlp_ratio)
    flops_attn_proj = 4 * seq_len * embed_dim ** 2
    flops_attn_score = 2 * seq_len ** 2 * embed_dim
    flops_mlp = 2 * (2 * seq_len * embed_dim * hidden)
    per_layer = flops_attn_proj + flops_attn_score + flops_mlp
    return int (per_layer * num_layers)

def estimate_flops_for_modified_vit(model, seq_len: int) -> int:
    """
    seq_len: the average number of tokens (1 CLS + L8 + L16 + L32)
    observed on the test subset -- pass in the average measured from 
    several real samples (the Modified ViT's token count VARIES per image,
    unlike the Baseline ViT which always has a fixed 197 tokens for 16x16
    patches on a 224x224 pixel image). This variablity is itself one of the
    efficiency arguments worth reporting: images with more homogenous
    regions produce fewer tokens and thus lower FLOPs -- the model adapts
    its own compute to image complexity.
    """
    cfg = model.backbone.encoder.layers[0]
    embed_dim = cfg.linear1.in_features
    num_heads = cfg.self_attn.num_heads
    num_layers = len(model.backbone.encoder.layers)
    mlp_ratio = cfg.linear1.out_features / embed_dim
    return estimate_transformer_flops(seq_len, embed_dim, num_layers, num_heads, mlp_ratio)

def measure_inference_time(model: nn.Module, loader, device: torch.device, num_batches: int = 20) -> float:
    """ 
    Average per-SAMPLE inference time (seconds), measured over the first 
    `num_batches` batches of `loader` (after skipping 3 warm-up batches that
    are not counted).
    """
    model.eval()
    times = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if isinstance(batch, dict):
                batch_data = {k: v.to(device) for k, v in batch.items()}
                batch_size = batch_data["labels"].size(0)
                forward = lambda: model(batch_data)
            else:
                images, labels = batch
                images = images.to(device)
                batch_size = images.size(0)
                forward = lambda: model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            forward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            if i >= 3:   # skip the first few batches as warm-up
                times.append(elapsed / batch_size)
            if i >= num_batches + 3:
                break
    return float(np.mean(times)) if times else float("nan")


# Scenario 3 - Dynamic lighting robustness

@dataclass
class LightingRobustnessResult:
    accuracy_normal: float
    accuracy_lighting_varied: float
    delta_accuracy: float

def lighting_robustness_test(
    model: nn.Module,
    normal_loader,
    lighting_varied_loader,
    device: torch.device,
) -> LightingRobustnessResult:
    """
    Compare accuracy on the normal test subset vs. the same subset
    passed through Dynamic Lighting Simulation (see
    `preprocessing.build_lighting_only_transform`). A smaller accuracy
    delta for the Final Optimized model compared to the Baseline model 
    is the succes indicator for this scenario.
    """
    acc_normal = compute_classification_metrics(model, normal_loader, device).accuracy
    acc_lighting = compute_classification_metrics(model, lighting_varied_loader, device).accuracy
    return LightingRobustnessResult(
        accuracy_normal=acc_normal,
        accuracy_lighting_varied=acc_lighting,
        delta_accuracy=acc_normal - acc_lighting,
    )

# Side-by-side comparison: standard (fixed-patch) ViT vs. the adaptive-patch 
# model -- produces the number for a Baseline-vs-Final-Optimized results table
# (accuracy, efficiency, lighting robustness).

@dataclass
class ModelComparisonReport:
    baseline_metrics: ClassificationMetrics
    optimized_metrics: ClassificationMetrics
    baseline_params: Dict[str, int]
    optimized_params: Dict[str, int]
    baseline_inference_time: float
    optimized_inference_time: float
    baseline_lighting: "LightingRobustnessResult"
    optimized_lighting: "LightingRobustnessResult"

    def print_table(self):
        def pct(x):
            return f"{x * 100:.2f}%"

        print(f"{'Metric':35s} {'Baseline (fixed patch)':>24s} {'Optimized (adaptive)':>24s}")
        print("-" * 85)
        print(f"{'Accuracy':35s} {pct(self.baseline_metrics.accuracy):>24s} {pct(self.optimized_metrics.accuracy):>24s}")
        print(f"{'Precision (macro)':35s} {pct(self.baseline_metrics.precision_macro):>24s} {pct(self.optimized_metrics.precision_macro):>24s}")
        print(f"{'Recall (macro)':35s} {pct(self.baseline_metrics.recall_macro):>24s} {pct(self.optimized_metrics.recall_macro):>24s}")
        print(f"{'F1 (macro)':35s} {pct(self.baseline_metrics.f1_macro):>24s} {pct(self.optimized_metrics.f1_macro):>24s}")
        print("-" * 85)
        print(f"{'Total parameters':35s} {self.baseline_params['total']:>24,d} {self.optimized_params['total']:>24,d}")
        print(f"{'Nonzero parameters':35s} {self.baseline_params['nonzero']:>24,d} {self.optimized_params['nonzero']:>24,d}")
        print(f"{'Inference time / sample (ms)':35s} {self.baseline_inference_time*1000:>24.3f} {self.optimized_inference_time*1000:>24.3f}")
        print("-" * 85)
        print(f"{'Accuracy (normal lighting)':35s} {pct(self.baseline_lighting.accuracy_normal):>24s} {pct(self.optimized_lighting.accuracy_normal):>24s}")
        print(f"{'Accuracy (varied lighting)':35s} {pct(self.baseline_lighting.accuracy_lighting_varied):>24s} {pct(self.optimized_lighting.accuracy_lighting_varied):>24s}")
        print(f"{'Delta accuracy (lighting)':35s} {pct(self.baseline_lighting.delta_accuracy):>24s} {pct(self.optimized_lighting.delta_accuracy):>24s}")
        print("(smaller delta = more robust to dynamic lighting)")

def compare_baseline_vs_optimized(
    baseline_model: nn.Module,
    optimized_model: nn.Module,
    baseline_test_loader,
    optimized_test_loader,
    baseline_lighting_loader,
    optimized_lighting_loader,
    device: torch.device,
) -> ModelComparisonReport:
    """
    Run all three evaluation scenarios (accuracy, efficiency, lighting
    robustness) for BTOH the standard `BaselineVisionTransformer` and the
    `ModifiedVisionTransformer` (ideally the pruned/fine-tuned final model),
    and return a report you can print as a table or drop straight into a 
    result section.

    `baseline_test_loader`/`baseline_lighting_loader` should be build with
    the plain-image collate (see `data.PlainImageDataset`), while 
    `optimized_test_loader`/`optimized_lighting_loader` should use
    `adaptive_collate_fn` -- the two models consume different batch shape.
    """
    baseline_metrics = compute_classification_metrics(baseline_model, baseline_test_loader, device)
    optimized_metrics = compute_classification_metrics(optimized_model, optimized_test_loader, device)

    baseline_params = count_parameters(baseline_model)
    optimized_params = count_parameters(optimized_model)

    baseline_time = measure_inference_time(baseline_model, baseline_test_loader, device)
    optimized_time = measure_inference_time(optimized_model, optimized_test_loader, device)

    baseline_lighting = lighting_robustness_test(baseline_model, baseline_test_loader, baseline_lighting_loader, device)
    optimized_lighting = lighting_robustness_test(optimized_model, optimized_test_loader, optimized_lighting_loader, device)

    report = ModelComparisonReport(
        baseline_metrics=baseline_metrics, optimized_metrics=optimized_metrics,
        baseline_params=baseline_params, optimized_params= optimized_params,
        baseline_inference_time=baseline_time, optimized_inference_time=optimized_time,
        baseline_lighting=baseline_lighting, optimized_lighting=optimized_lighting
    )
    report.print_table()
    return report