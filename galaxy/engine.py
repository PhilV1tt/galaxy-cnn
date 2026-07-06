"""Training loop, evaluation, and fit with early stopping on macro F1."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import NUM_CLASSES, get_device
from .metrics import macro_f1


def class_weights(labels, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Weights = inverse frequency, normalized. For CrossEntropyLoss(weight=)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, device=None):
    """Returns (mean loss, y_true, y_pred). Sets the model to eval()."""
    device = device or get_device()
    model.eval()
    total_loss, n = 0.0, 0
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        ps.append(out.argmax(1).cpu())
        ys.append(yb.cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    return total_loss / n, y_true, y_pred


def train_one_epoch(model, loader, optimizer, criterion, device=None) -> float:
    device = device or get_device()
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n


def fit(
    model,
    train_loader,
    val_loader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    criterion=None,
    optimizer=None,
    scheduler: str | None = "cosine",
    patience: int = 15,
    device=None,
    verbose: bool = True,
):
    """Trains and keeps the best weights according to the validation macro F1.

    Returns (best_model, history) where history is a dict of lists per epoch.
    """
    device = device or get_device()
    model.to(device)
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = optimizer or torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    sched = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_f1_macro": [], "val_acc": []}
    best_f1, best_state, best_epoch, since = -1.0, None, -1, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        f1 = macro_f1(y_true, y_pred)
        acc = float((y_true == y_pred).mean())
        if sched is not None:
            sched.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_f1_macro"].append(f1)
        history["val_acc"].append(acc)

        if f1 > best_f1:
            best_f1, best_state, best_epoch, since = f1, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            since += 1

        if verbose:
            print(
                f"epoch {epoch:2d} | train {tr_loss:.3f} | val {val_loss:.3f} "
                f"| val F1 {f1:.3f} | val acc {acc:.3f} | {time.time()-t0:.1f}s"
            )
        if since >= patience:
            if verbose:
                print(f"early stopping (best F1 {best_f1:.3f} @ epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_f1_macro"] = best_f1
    return model, history
