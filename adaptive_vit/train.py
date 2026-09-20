"""
train.py

Model training loop.

Loss: standard cross-entropy (L_CE = -sum_c y_c log(y_hat_c)), applied via
`nn.CrossEntropyLoss` (numerically stable log-softmax + NLL, equivalent to 
classification-head logits + softmax + NLL).

Optimizer: AdamW + cosine-annealing LR scheduler, batch size 32, early stopping
on validation loss with configurable patience.

Functions here are generic: they work for `ModifiedVisionTransformer`
(batch is a dict from `adaptive_collate_fn`) as well as `BaselineVisionTransformer`
(batch is a (images, labels) tuple from PyTorch's default collate) -- detected
automatically from the batch type.
"""

import copy
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import TrainConfig

def _forward_and_labels(model: nn.Module, batch, device: torch.device):
    """
    Auto-detect the batch type (dict for the Modified ViT, tuple for the Baseline ViT)
    and return (logits, labels).
    """
    if isinstance(batch, dict):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        logits = model(batch)
    else:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
    return logits, labels

@dataclass
class EpochResult:
    loss: float
    accuracy: float

def run_epoch(
    model: nn.Module,
    loader, 
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip_norm: Optional[float] = None
) -> EpochResult:
    """
    One training epoch (if `optimizer` is given) or one evaluation pass
    (if `optimizer` is None).
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_correct, total_samples = 0.0, 0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            logits, labels = _forward_and_labels(model, batch, device)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return EpochResult(loss=total_loss / total_samples, accuracy=total_correct / total_samples)

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    verbose: bool = True,
):
    """
    Full training loop with cosine annelaing + early stopping 
    (patience = `cfg.early_stopping_patinece` epochs, based on validation loss).

    `class_weights`: optional FloatTensor [num_classes] -- set this if 
    `data.validate_dataset()`'s repor shows significant class imbalance 
    (`classes_exceeding_tolerance`), so CrossEntropyLoss up-weight minory classes.

    Returns: (model_with_best_weight, history_dict)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        train_result = run_epoch(model, train_loader, device, criterion, optimizer, cfg.grad_clip_norm)
        val_result = run_epoch(model, val_loader, device, criterion, optimizer=None)
        scheduler.step()

        history["train_loss"].append(train_result.loss)
        history["train_acc"].append(train_result.accuracy)
        history["val_loss"].append(val_result.loss)
        history["val_acc"].append(val_result.accuracy)

        if verbose:
            print(
                f"[epoch {epoch:03d}/{cfg.num_epochs}] "
                f"train_loss={train_result.loss:.4f} train_acc={train_result.accuracy:.4f} | "
                f"val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s"
            )

        if val_result.loss < best_val_loss - 1e-6:
            best_val_loss = val_result.loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                if verbose:
                    print(
                        f"[train.py] Early stopping at epoch {epoch} "
                        f"(no val_loss improvement for "
                        f"{cfg.early_stopping_patience} epochs)."
                    )
                break

    model.load_state_dict(best_state)
    return model, history