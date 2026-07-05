"""Chargement, split, normalisation et augmentation de Galaxy10 DECaLS.

Pipeline :
  - décodage HF -> tenseurs uint8 [N,3,res,res] mis en cache par résolution ;
  - split validation stratifié 15 % pris dans le train (le test officiel reste intact) ;
  - normalisation par mean/std calculés sur le train uniquement ;
  - augmentation (rotation 0-360, flips H/V, léger jitter) sur le train seulement.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .config import HF_DATASET, SEED, data_dir


# --- Décodage HF -> tenseurs uint8 ------------------------------------------
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
    """Charge un split complet (train/test) en tenseurs uint8, avec cache disque."""
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
    """Charge n exemples par streaming (smoke test, sans télécharger tout le parquet)."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=split, streaming=True)
    return _decode(ds, res, n=n)


# --- Normalisation ----------------------------------------------------------
def compute_mean_std(x_uint8: torch.Tensor):
    """mean/std par canal sur des images uint8 (renvoie des floats en [0,1])."""
    x = x_uint8.float().div_(255.0)
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3))
    return mean, std


# --- Transforms -------------------------------------------------------------
def build_transforms(mean, std, train: bool):
    """train=True -> augmentation physique (rotations/flips). Sinon normalisation seule."""
    mean = tuple(float(m) for m in mean)
    std = tuple(float(s) for s in std)
    if train:
        return v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(180),  # +/-180 deg : une galaxie n'a pas d'orientation privilégiée
            v2.ColorJitter(brightness=0.1, contrast=0.1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ])
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])


class GalaxyDataset(Dataset):
    """Tenseur uint8 en mémoire + transform appliquée à la volée."""

    def __init__(self, x_uint8: torch.Tensor, y: torch.Tensor, transform):
        self.x = x_uint8
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.y[i]


# --- Assemblage -------------------------------------------------------------
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
    """Retourne (train_ds, val_ds, test_ds, (mean, std)).

    smoke_n : si vrai (booléen ou entier), charge le petit split test officiel (1800
    images, toutes les classes) et le découpe 60/20/20, uniquement pour vérifier que
    le pipeline tourne sans télécharger les 2,3 Go du train. L'expérience réelle
    utilise smoke_n=None : le test HF reste le test final, la validation vient du train.
    mean/std : passer les stats ImageNet pour le ResNet pré-entraîné ; None = stats train.
    augment=False désactive l'augmentation (ablation).
    """
    if smoke_n is not None:
        x, y = load_split("test", res, cache=cache)  # 1800 imgs, 10 classes garanties
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
