"""Loading, split, normalization and augmentation of Galaxy10 DECaLS.

Pipeline:
  - HF decoding -> uint8 tensors [N,3,res,res] cached by resolution;
  - stratified 15% validation split taken from train (official test stays intact);
  - normalization by mean/std computed on train only;
  - augmentation (0-360 rotation, H/V flips, light jitter) on train only.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .config import HF_DATASET, SEED, data_dir


# --- HF decoding -> uint8 tensors -------------------------------------------
def _decode(examples_iter, res: int, n: int | None = None):
    xs, ys = [], []
    for i, ex in enumerate(examples_iter):
        if n is not None and i >= n:
            break
        img: Image.Image = ex["image"].convert("RGB").resize((res, res), Image.BILINEAR)
        xs.append(torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).contiguous())
        ys.append(int(ex["label"]))
    x = torch.stack(xs)  # uint8 [N,3,res,res]
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def load_split(split: str, res: int, cache: bool = True):
    """Load a full split (train/test) as uint8 tensors, with disk cache."""
    from datasets import load_dataset

    cache_path = data_dir() / "cache" / f"{split}_{res}.pt"
    if cache and cache_path.exists():
        blob = torch.load(cache_path)
        return blob["x"], blob["y"]
    ds = load_dataset(HF_DATASET, split=split)
    x, y = _decode(ds, res)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": x, "y": y}, cache_path)
    return x, y


def load_stream(split: str, res: int, n: int):
    """Load n examples by streaming (smoke test, without downloading the whole parquet)."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=split, streaming=True)
    return _decode(ds, res, n=n)


# --- Normalization ----------------------------------------------------------
def compute_mean_std(x_uint8: torch.Tensor):
    """Per-channel mean/std on uint8 images (returns floats in [0,1])."""
    x = x_uint8.float().div_(255.0)
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3))
    return mean, std


# --- Transforms -------------------------------------------------------------
def build_transforms(mean, std, train: bool):
    """train=True -> physical augmentation (rotations/flips). Otherwise normalization only."""
    mean = tuple(float(m) for m in mean)
    std = tuple(float(s) for s in std)
    if train:
        return v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(180),  # +/-180 deg: a galaxy has no preferred orientation
            v2.ColorJitter(brightness=0.1, contrast=0.1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ])
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])


class GalaxyDataset(Dataset):
    """uint8 tensor in memory + transform applied on the fly."""

    def __init__(self, x_uint8: torch.Tensor, y: torch.Tensor, transform):
        self.x = x_uint8
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.y[i]


# --- Assembly ---------------------------------------------------------------
def get_datasets(
    res: int = 64,
    val_frac: float = 0.15,
    seed: int = SEED,
    smoke_n: int | None = None,
    cache: bool = True,
    mean=None,
    std=None,
    augment: bool = True,
):
    """Returns (train_ds, val_ds, test_ds, (mean, std)).

    smoke_n: if truthy (boolean or integer), loads the small official test split (1800
    images, all classes) and splits it 60/20/20, only to check that the pipeline runs
    without downloading the 2.3 GB of train. The real experiment uses smoke_n=None: the
    HF test stays the final test, validation comes from train.
    mean/std: pass ImageNet stats for the pretrained ResNet; None = train stats.
    augment=False disables augmentation (ablation).
    """
    if smoke_n is not None:
        x, y = load_split("test", res, cache=cache)  # 1800 imgs, 10 classes guaranteed
        if isinstance(smoke_n, int) and smoke_n < len(y):
            keep, _ = train_test_split(
                np.arange(len(y)), train_size=smoke_n, stratify=y.numpy(), random_state=seed
            )
            x, y = x[keep], y[keep]
        idx = np.arange(len(y))
        tr, tmp = train_test_split(idx, test_size=0.4, stratify=y.numpy(), random_state=seed)
        val, te = train_test_split(tmp, test_size=0.5, stratify=y.numpy()[tmp], random_state=seed)
        xtr, ytr, xval, yval, xte, yte = x[tr], y[tr], x[val], y[val], x[te], y[te]
    else:
        xall, yall = load_split("train", res, cache=cache)
        xte, yte = load_split("test", res, cache=cache)
        tr, val = train_test_split(
            np.arange(len(yall)), test_size=val_frac, stratify=yall.numpy(), random_state=seed
        )
        xtr, ytr, xval, yval = xall[tr], yall[tr], xall[val], yall[val]

    if mean is None or std is None:
        mean, std = compute_mean_std(xtr)

    train_ds = GalaxyDataset(xtr, ytr, build_transforms(mean, std, train=augment))
    val_ds = GalaxyDataset(xval, yval, build_transforms(mean, std, train=False))
    test_ds = GalaxyDataset(xte, yte, build_transforms(mean, std, train=False))
    return train_ds, val_ds, test_ds, (mean, std)


def make_loaders(train_ds, val_ds, test_ds, batch_size: int = 128, num_workers: int = 0):
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )
