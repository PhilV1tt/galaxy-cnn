"""Galaxy10 project package: morphological classification of galaxies with a CNN."""
from __future__ import annotations

from . import config, data, engine, gradcam, metrics, models, viz
from .config import CLASS_NAMES, NUM_CLASSES, get_device, set_seed

__all__ = [
    "config", "data", "models", "engine", "metrics", "viz", "gradcam",
    "CLASS_NAMES", "NUM_CLASSES", "get_device", "set_seed",
]
