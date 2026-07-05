"""Project figures. Each function saves into figures/ and returns the path."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import CLASS_NAMES, figures_dir


def _save(fig, name: str) -> Path:
    path = figures_dir() / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _to_hwc(img_uint8: torch.Tensor) -> np.ndarray:
    return img_uint8.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def plot_samples_grid(x_uint8, y, class_names=CLASS_NAMES, name="samples.png"):
    """One image per class (first one found)."""
    y = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
    for c, ax in enumerate(axes.ravel()):
        idx = np.where(y == c)[0]
        if len(idx):
            ax.imshow(_to_hwc(x_uint8[idx[0]]))
        ax.set_title(f"{c} · {class_names[c]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Galaxy10 DECaLS: one example per class", fontsize=12)
    return _save(fig, name)


def plot_class_distribution(y, class_names=CLASS_NAMES, name="distribution.png"):
    y = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    counts = np.bincount(y, minlength=len(class_names))
    order = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(counts)), counts[order], color="#4a86e8")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([class_names[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("number of images")
    ax.set_title(f"Class distribution (imbalance, factor {counts.max()/counts.min():.1f})")
    return _save(fig, name)


def plot_augmentation(img_uint8, mean, std, n=6, name="augment.png"):
    """The same galaxy augmented n times (rotations/flips/jitter), for illustration."""
    from torchvision.transforms import v2

    disp = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(180),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.ToDtype(torch.float32, scale=True),
    ])
    fig, axes = plt.subplots(1, n + 1, figsize=(2.0 * (n + 1), 2.2))
    axes[0].imshow(_to_hwc(img_uint8))
    axes[0].set_title("original", fontsize=9)
    axes[0].axis("off")
    for k in range(1, n + 1):
        axes[k].imshow(disp(img_uint8).permute(1, 2, 0).clamp(0, 1).numpy())
        axes[k].set_title(f"aug {k}", fontsize=9)
        axes[k].axis("off")
    fig.suptitle("Augmentation (train only): physical invariances", fontsize=11)
    return _save(fig, name)


def plot_curves(history, name="courbes.png"):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    a.plot(ep, history["train_loss"], label="train")
    a.plot(ep, history["val_loss"], label="val")
    a.set_xlabel("epoch"); a.set_ylabel("loss"); a.set_title("Loss"); a.legend()
    b.plot(ep, history["val_f1_macro"], label="F1 macro", color="#16a765")
    b.plot(ep, history["val_acc"], label="accuracy", color="#e07798")
    b.set_xlabel("epoch"); b.set_ylabel("val"); b.set_title("Validation"); b.legend()
    if history.get("best_epoch", -1) > 0:
        for ax in (a, b):
            ax.axvline(history["best_epoch"], ls="--", c="gray", lw=1)
    return _save(fig, name)


def plot_confusion(cm, class_names=CLASS_NAMES, name="confusion.png", normalized=True):
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fmt = ".2f" if normalized else "d"
    thr = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=7)
    ax.set_title("Confusion matrix" + (" (normalized)" if normalized else ""))
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, name)


def plot_perclass_f1(per_class, name="perclass.png"):
    """per_class: DataFrame index=class, column 'f1'."""
    f1 = per_class["f1"].values
    names = list(per_class.index)
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(range(len(f1)), f1[order], color="#43d692")
    ax.set_yticks(range(len(f1)))
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel("F1"); ax.set_xlim(0, 1)
    ax.set_title("Per-class F1 (test)")
    return _save(fig, name)


def plot_scale_comparison(results: dict, metric="f1_macro", name="echelle.png"):
    """results: {model_name: {metric: value}} -> baseline -> CNN -> ResNet bars."""
    names = list(results.keys())
    vals = [results[k][metric] for k in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#cccccc", "#999999", "#4a86e8", "#16a765"][: len(names)]
    ax.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel(metric); ax.set_ylim(0, 1)
    ax.set_title("What does convolution buy? (test)")
    return _save(fig, name)
