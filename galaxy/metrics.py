"""Métriques d'évaluation (numériques). Les figures sont dans viz.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import CLASS_NAMES


def summary_metrics(y_true, y_pred) -> dict:
    """Accuracy + precision/recall/F1 en macro et pondéré."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f_macro,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f_w,
    }


def per_class_df(y_true, y_pred, class_names=CLASS_NAMES) -> pd.DataFrame:
    """Precision/recall/F1/support par classe."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    return pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": s}, index=class_names
    )


def confusion(y_true, y_pred, normalize: str | None = None) -> np.ndarray:
    """Matrice de confusion. normalize='true' pour des lignes en proportion."""
    n = len(CLASS_NAMES)
    return confusion_matrix(y_true, y_pred, labels=range(n), normalize=normalize)


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
