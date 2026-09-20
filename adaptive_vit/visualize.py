"""
visualize.py

Plotting utilities for your result section -- training curves, confusion 
matrix, a Baseline-vs-Optimized comparison chart, the pruning-ratio search
trajectory, and (the most distinctive one) a visualization of the adaptive 
patch-size map itself overlaid on an actual image.

Every function returns the matplotli `Figure` it created (so you can
further tweak it, e.g. `fig.suptitle(...)`) AND optionally saves it to 
`save_patch` if given. None of these funciton call `plt.show()` 
themselves -- call that yourself in a notebook cell, or just save to 
a file.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

def _required_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "visualize.py needs matplotlib. Install it with `pip install matplotlib seaborn`."
        ) from e


# Training curve

def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    `history`: the dict returned by `train.train_model` (keys
    "train_loss", "train_acc", "val_loss", "val_acc")/ Produces a 1x2
    figure: loss curves on the left, accuracy curves on the right.
    """
    plt = _required_matplotlib()
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training vs. Validation loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training vs. Validation accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

# Confusion matrix

def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
):
    """
    Build & plot a confusion matrix heatmap from prediciton arrays.

    Get `y_true`/`y_pred` by running your model over the test loader once
    and collecting predictions yourself (or adapt 
    `evaluate.compute_classification_metrics` to also return them -- it
    currently only returns aggregate metrics, not raw predictions).

    With many classes (e.g. 29), pass `figsize=(14, 12)` or similar so
    labels stay readable.
    """
    plt = _required_matplotlib()
    from sklearn.metrics import confusion_matrix
    # Import seaborn dynamically so this module remains usable when the
    # optional plotting dependency is not installed.
    import importlib

    try:
        sns = importlib.import_module("seaborn")
    except ImportError as exc:
        raise ImportError(
            "plot_confusion_matrix needs seaborn. Install it with "
            "`pip install seaborn`."
        ) from exc

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
        fmt = ".2f"
    else:
        fmt = "d"

    figsize = figsize or (max(8, len(class_names) * 0.4), max(6, len(class_names) * 0.4))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=len(class_names) <=15, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True,
        square=True,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix" + (" (normalized)" if normalize else ""))
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

# Baseline vs. Optimized comparison bars

def plot_model_comparison(report, save_path: Optional[str] = None):
    """
    `report`: a `eavluate.ModelComparisonRepor` (the return value of 
    `evaluate.compare_baseline_vs_optimized`). Produces a 1x3 figure:
    accuracy, parameter count, and lighting-robustness delta, Baseline 
    vs. Optimized side by side
    """
    plt = _required_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ["Baseline\n(fixed patch)", "Optimized\n(adaptive)"]
    colors = ["#9aa5b1", "#2b6cb0"]

    accs = [report.baseline_metrics.accuracy * 100, report.optimized_metrics.accuracy * 100]
    axes[0].bar(labels, accs, color=colors)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Test accuracy")
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.5, f"{v:.2f}%", ha="center")
    axes[0].set_ylim(0, 105)

    params = [report.baseline_params["total"] / 1e6, report.optimized_params["nonzero"] / 1e6]
    axes[1].bar(labels, params, color=colors)
    axes[1].set_ylabel("Parameters (milions)")
    axes[1].set_title("Baseline (total) vs. \nOptmizied (nonzero after pruning)")
    for i, v in enumerate(params):
        axes[1].text(i, v + max(params) * 0.2, f"{v:.2f}M", ha="center")

    deltas = [report.baseline_lighting.delta_accuracy * 100, report.optimized_lighting.delta_accuracy * 100]
    axes[2].bar(labels, deltas, color=colors)
    axes[2].set_ylabel("Accuracy delta(%)")
    axes[2].set_title("Accuracy drop under\nvaried lighting (lower = more robust)")
    for i, v in enumerate(deltas):
        axes[2].text(i, v + max(max(deltas), 0.1) * 0.2, f"{v:.2f}%", ha="center")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

# Pruning-ratio search trajectory

def plot_pruning_search(history: List, save_path: Optional[str] = None):
    """
    `history`: `PruningResult.history` from
    `pruning.iterative_prune_and_finetune` -- a list of
    `pruning.PruningIterationLog`. Plots validation accuracy against the
    prune ratio tried at each iteration, marking the accepted one.
    """
    plt = _required_matplotlib()

    ratios = [h.achieved_ratio * 100 for h in history]
    accs = [h.val_accuracy * 100 for h in history]
    accepted = [h.accepted for h in history]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ratios, accs, "o-", color="#2b6cb0", zorder=1)
    for r, a, ok in zip(ratios, accs, accepted):
        ax.scatter(r, a, s=140 if ok else 70,
                    color="#38a169" if ok else "#e53e3e",
                    edgecolor="black", zorder=2)
    ax.set_xlabel("Achieved prune ratio (%)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Pruning-ratio search")
    ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#38a169", markeredgecolor="black", markersize=10, label="Accepted (within tolerance)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e53e3e", markeredgecolor="black", markersize=8, label="Rejected (exceeded tolerance)"),
    ]
    ax.legend(handles=legend_elems, loc="best")
    for h in history:
        ax.annotate(f"it.{h.iteration}", (h.achieved_ratio * 100, h.val_accuracy * 100),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# Adaptive patch-size map overlay -- the most distinctive figure for this
# study: shows WHERE the model chose small/medium/large patches on an 
# actual image, directly visualizing the core contribution (ICS-driven
# adaptive granularity).

def visualize_patch_size_map(
    image_np: np.ndarray,
    patch_size_map: np.ndarray,
    region_size: int = 32,
    save_path: Optional[str] = None,
):
    """
    `image_np`: (H, W, 3) array in [0,1] or [0.255] (e.g. the
    pre-normalization tensor from `data.ImageListDataset`, permuted to 
    HWC and converted to numpy).
    `patch_size_map`: the (n_rows, n_cols) array from 
    `ics.compute_patch_size_map_for_image` (values in {8, 16, 32}).

    Produces a 1x2 figure: the original image on the left, and the same
    image on the right with a color-coded grid overlay showing which patch
    size (8/16/32) was choosen for each region -- small (red) = high
    complexity/detail kept, large (blue) Homogeneous/efficient.
    """
    plt = _required_matplotlib()
    import matplotlib.patches as mpatches

    img_display = image_np.astype(np.float64)
    if img_display.max() > 1.5 :    # looks liek a [0,255] image
        img_display = img_display / 255.0
    img_display = np.clip(img_display, 0, 1)

    color_map = {8: "#e53e3e", 16: "#ecc94b", 32: "#3182ce"}  # small=red, medium=yellow, large=blue
    label_map = {8: "8x8 (high complexity)", 16: "16x16 (medium)", 32: "32x32 (homogeneous)"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    axes[0].imshow(img_display)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(img_display)
    n_rows, n_cols = patch_size_map.shape
    for rr in range(n_rows):
        for rc in range(n_cols):
            size = int(patch_size_map[rr, rc])
            rect = mpatches.Rectangle(
                (rc * region_size, rr * region_size), region_size, region_size,
                linewidth=1.2, edgecolor=color_map[size], facecolor=color_map[size], alpha=0.35
            )
            axes[1].add_patch(rect)
    axes[1].set_title("Adaptive patch-size map")
    axes[1].axis("off")

    legend_handles = [mpatches.Patch(color=color_map[s], label=label_map[s], alpha=0.6) for s in (8, 16, 32)]
    axes[1].legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.83), ncol=1, fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig