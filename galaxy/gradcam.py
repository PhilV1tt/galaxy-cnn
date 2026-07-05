"""Grad-CAM à la main (Selvaraju et al., 2017).

Principe : pour la classe cible, on prend le gradient du score par rapport aux
feature maps de la dernière conv, on en fait une moyenne spatiale (importance de
chaque canal), on pondère les feature maps, ReLU, puis on redimensionne à la taille
de l'image. La carte montre *où* le réseau regarde.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .config import CLASS_NAMES, figures_dir, get_device
from .data import build_transforms


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx: int | None = None):
        """x : [1,3,H,W] normalisé. Retourne (heatmap [H,W] en [0,1], classe)."""
        self.model.eval()
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1).item())
        self.model.zero_grad()
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # [1,C,1,1]
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))  # [1,1,h,w]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx


def _overlay(img_hwc_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = plt.get_cmap("jet")(cam)[..., :3]  # [H,W,3] float
    base = img_hwc_uint8.astype(np.float32) / 255.0
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def plot_gradcam_grid(
    model,
    target_layer,
    images_uint8,
    mean,
    std,
    y_true=None,
    class_names=CLASS_NAMES,
    device=None,
    name="gradcam.png",
):
    """Pour chaque image (uint8 [3,H,W]) : original + carte Grad-CAM superposée."""
    device = device or get_device()
    model.to(device)
    cam = GradCAM(model, target_layer)
    tfm = build_transforms(mean, std, train=False)

    m = len(images_uint8)
    fig, axes = plt.subplots(2, m, figsize=(2.4 * m, 5.0))
    if m == 1:
        axes = axes.reshape(2, 1)
    for k in range(m):
        img = images_uint8[k]
        x = tfm(img).unsqueeze(0).to(device)
        heat, pred = cam(x)
        hwc = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        axes[0, k].imshow(hwc)
        title = class_names[pred]
        if y_true is not None:
            t = int(y_true[k])
            ok = "OK" if t == pred else f"vrai {class_names[t]}"
            title = f"pred {class_names[pred]}\n{ok}"
        axes[0, k].set_title(title, fontsize=8)
        axes[0, k].axis("off")
        axes[1, k].imshow(_overlay(hwc, heat))
        axes[1, k].axis("off")
    fig.suptitle("Grad-CAM : où le réseau regarde", fontsize=12)
    path = figures_dir() / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
