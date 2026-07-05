"""Constants, device and paths for the Galaxy10 project."""
from __future__ import annotations

import os
from pathlib import Path

import torch

# --- Galaxy10 DECaLS classes (order = label 0..9) ---------------------------
CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
]
# Official counts (astroNN), useful for the EDA and a consistency check.
CLASS_COUNTS = [1081, 1853, 2645, 2027, 334, 2043, 1829, 2628, 1423, 1873]
NUM_CLASSES = len(CLASS_NAMES)

HF_DATASET = "matthieulel/galaxy10_decals"
SEED = 42

# ImageNet stats (for the pretrained ResNet backbone).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Paths ----------------------------------------------------------------
IN_COLAB = "google.colab" in os.sys.modules or os.environ.get("COLAB_RELEASE_TAG") is not None


def project_root() -> Path:
    """Project root, robust both locally and on Colab."""
    if IN_COLAB:
        p = Path("/content/galaxy-cnn")
        p.mkdir(parents=True, exist_ok=True)
        return p
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        # Run outside a module (inlined notebook cells): fall back to the cwd.
        return Path.cwd()


def data_dir() -> Path:
    d = project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir() -> Path:
    d = project_root() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoints_dir() -> Path:
    d = project_root() / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Device -----------------------------------------------------------------
def get_device() -> torch.device:
    """CUDA (Colab) > MPS (Mac) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
