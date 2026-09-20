"""
pruning.py

Structured Magnitude-Based Pruning and post-pruning validation.

Base formulas:
    s_i = |w_i|
    m_i = 1 if s_i >= threshold, else 0
    w_i' = w_i * m_i

"Structured" means the unit being pruned is an ENTIRE NEURON in each
Transformer-encoder MLP block (not an individual, element-wise weight as in
unstructured pruning) -- so that pruning translate into a real potential 
speed-up once those neurons are actually removed, rather than just random 
sparsity. The per-scaler-weight magnitude score aboce is extended to a 
per-structural-unit score: s_i = |W_in[i, :]||_1, i.e. the L1 norm of a 
neuron's entire incoming weight vector -- a natural generalization of the base
formula to the "structural unit" level, consistent with the "magnitude-based"
part of the method's name.

Pruned unit: a neuron in `linear1` (the d_model -> d_ff projection) of each
`nn.TransformerEncoderLayer`. A pruned neuron is also "disconnected" from 
`linear2` (the d_ff -> d_model projection), so its contribution to the residual
stream is truly zero rather than leaking through via bias terms.
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from .config import PruningConfig, TrainConfig
from .train import _forward_and_labels, run_epoch

LayerMask = torch.Tensor    # Booltensor [d_ff], True = neuron KEPT

def _get_encoder_layers(model: nn.Module) -> List[nn.TransformerEncoderLayer]:
    """
    Collect ever nn.TransformerEncoderLayer from the shared backbone
    (used by both ModifiedVisionTransformer & BaselineVisionTransformer,
    see vit_model.py).
    """
    return list(model.backbone.encoder.layers)

def compute_prune_mask(model: nn.Module, ratio: float) -> List[LayerMask]:
    """
    Compute the magnitude score s_i = ||W1[i, :]||_1 for EVERY MLP neuron
    in EVERY encoder layer, pool them into one global distribution, then 
    pick a `threshold` such that roughly `ratio` fraction of neurons 
    (globally, across all layers) fall below it -> mask m_i = 0 (pruned).
    Neurons with s_i >= threshold are kept (m_i = 1) following the base
    formula above.

    Returns: a list of per-layer mask (same order as `_get_encoder_layers`).  
    """
    layers = _get_encoder_layers(model)
    all_scores = []
    per_layer_scores = []

    for layer in layers:
        w_in = layer.linear1.weight.detach()        # [d_ff, d_model]
        scores = w_in.abs().sum(dim=1)              # s_i = ||W_in[i,:]||_1, shape
        per_layer_scores.append(scores)
        all_scores.append(scores)

    all_scores_flat = torch.cat(all_scores)
    threshold = torch.quantile(all_scores_flat, ratio)  # bottom `ratio` fraction -> pruned

    masks = [scores >= threshold for scores in per_layer_scores]     # True = keep
    return masks

def apply_structured_pruning(model: nn.Module, masks: List[LayerMask]) -> None:
    """
    Apply the mask to `linear1` (rows/neurons) and the matching `linear2`
    columns -- w_1' = w_i * m_i, in place.
    """
    layers = _get_encoder_layers(model)
    assert len(layers) == len(masks), "Number of maksks must match the number of encoder"

    with torch.no_grad():
        for layer, keep_mask in zip(layers, masks):
            prune_idx = ~keep_mask
            layer.linear1.weight[prune_idx, :] = 0.0
            layer.linear1.bias[prune_idx] = 0.0
            layer.linear2.weight[:, prune_idx] = 0.0

def reapply_masks(model: nn.Module, masks: List[LayerMask]) -> None:
    """
    Call this AFTER EVERY `optimizer.step()` during fine-tuning, so
    pruned neurons don't "come back to life" due to gradient updates
    (weight decay/ momentum could otherwise push zeroed weights away from
    zero again unless the mask is enfoce continuously).
    """
    apply_structured_pruning(model, masks)

def structured_magnitude_pruning(model: nn.Module, ratio: float) -> List[LayerMask]:
    """
    Convenience function: compute masks then apply them immediately
    (in place on `model`). Returns the masks so they can be reused by
    `reapply_masks` throughout fine-tuning.
    """
    masks = compute_prune_mask(model, ratio)
    apply_structured_pruning(model, masks)
    return masks

def achieved_prune_ratio(masks: List[LayerMask]) -> float:
    """
    Compute the ACTUAL pruning ratio (r = N_pruned / N_total) from
    the masks that were really applied.
    """
    total = sum(m.numel() for m in masks)
    pruned = sum((~m).sum().item() for m in masks)
    return pruned / total


# Post-prunign fine-tuning, with the mask kept enforced

def finetune_with_mask(
    model: nn.Module,
    masks: List[LayerMask],
    train_loader,
    val_loader,
    cfg: PruningConfig,
    device: torch.device,
    verbose: bool = True,
):
    """
    Low-learning-rate fine-tuning after pruning, with `reaply_masks`
    called after every `optimizer.step()` so pruned neurons stay at zero
    throughout fine-tuning.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.finetune_lr)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg.finetune_epochs + 1):
        model.train()
        for batch in train_loader:
            logits, labels = _forward_and_labels(model, batch, device)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            reapply_masks(model, masks)  # lock the mask after every update

        val_result = run_epoch(model, val_loader, device, criterion, optimizer=None)
        if verbose:
            print(f"   [fine-tune epoch {epoch}/{cfg.finetune_epochs}] "
                  f"val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f}")

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model

@dataclass
class PruningIterationLog:
    """One row of the pruning-ratio search trajectory, kept so the whole
    search history can be plotted (see `visualize.plot_pruning_search`),
    not just the final accepted iteration."""
    iteration: int
    ratio_tried: float
    achieved_ratio: float
    val_accuracy: float
    accuracy_drop: float
    accepted: bool

@dataclass
class PruningResult:
    model: nn.Module
    masks: List[LayerMask]
    final_ratio: float
    val_accuracy: float
    baseline_accuracy: float
    accuracy_drop: float
    num_iterations: int
    history: List[PruningIterationLog]

def iterative_prune_and_finetune(
    trained_model: nn.Module,
    baseline_accuracy: float,
    train_loader,
    val_loader,
    cfg: PruningConfig,
    device: torch.device,
    verbose: bool = True,
) -> PruningResult:
    """
    Main pruning loop: prune -> fine-tune -> compare accuracy against
    `baseline_accuracy` (the trained model's accuracy BEFORE pruning) 
    -> if the accuracy drop exceeds `cfg.accuracy_drop_tolerance`, reduced
    the prune ratio by `cfg.ratio_step` and REPEAT the whole process 
    (always restarting from the original, unpruned `trained_model` -- not
    continuing from the already-pruned candidate).
    """
    ratio = cfg.initial_prune_ratio
    result = None
    history: List[PruningIterationLog] = []

    for iteration in range(1, cfg.max_prune_iterations + 1):
        if verbose:
            print(f"[pruning.py] Iteration {iteration}: trying prune ratio {ratio:.2%}")

        candidate = copy.deepcopy(trained_model)
        masks = structured_magnitude_pruning(candidate, ratio)
        candidate = finetune_with_mask(candidate, masks, train_loader, val_loader, cfg, device, verbose)

        criterion = nn.CrossEntropyLoss()
        val_result = run_epoch(candidate, val_loader, device, criterion, optimizer=None)
        drop = baseline_accuracy - val_result.accuracy
        achieved = achieved_prune_ratio(masks)
        accepted = drop <= cfg.accuracy_drop_tolerance

        if verbose:
            print(f" -> val_acc={val_result.accuracy:.4f} (baseline={baseline_accuracy:.4f}) "
                  f"drop={drop:.4f}, tolerance={cfg.accuracy_drop_tolerance:.4f}")

        history.append(PruningIterationLog(
            iteration=iteration, ratio_tried=ratio, achieved_ratio=achieved,
            val_accuracy=val_result.accuracy, accuracy_drop=drop, accepted=accepted,
        ))

        result = PruningResult(
            model=candidate, masks=masks, final_ratio=achieved,
            val_accuracy=val_result.accuracy, baseline_accuracy=baseline_accuracy,
            accuracy_drop=drop, num_iterations=iteration, history=history,
        )

        if accepted:
            if verbose:
                print(f"[pruning.py] Tolerance criterion satisfied at iteration {iteration}.")
            break

        ratio = max(ratio - cfg.ratio_step, 0.0)
        if ratio <= 0.0:
            if verbose:
                print("[pruning.py] Prune ratio reached 0%, stopping iterations.")
            break

    return result