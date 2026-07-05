"""Les modèles du projet : baselines sans convolution, CNN maison, ResNet18.

L'histoire du projet est l'échelle de trois niveaux :
  1. baselines (régression logistique, MLP) aveugles à la structure spatiale ;
  2. CNN maison (convolution = partage de poids + équivariance par translation) ;
  3. transfer learning (ResNet18 pré-entraîné ImageNet).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import NUM_CLASSES


class LogisticRegression(nn.Module):
    """Baseline 1 : régression logistique multinomiale sur pixels aplatis.

    Une seule couche linéaire. Aucune notion de localité : chaque pixel est une
    variable indépendante. Volontairement médiocre.
    """

    def __init__(self, in_channels: int = 3, image_size: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_features = in_channels * image_size * image_size
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class MLP(nn.Module):
    """Baseline 2 : perceptron multicouche sur pixels aplatis.

    Deux couches cachées + ReLU + dropout. Plus expressif que la régression
    logistique, mais toujours aveugle à la structure 2D.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 64,
        hidden=(512, 256),
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
    ):
        super().__init__()
        in_features = in_channels * image_size * image_size
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _conv_block(in_c: int, out_c: int, use_bn: bool = True) -> nn.Sequential:
    """Bloc [Conv 3x3 pad 1 -> (BatchNorm) -> ReLU -> MaxPool 2].

    La conv 3x3 pad 1 conserve la taille spatiale ; le pooling la divise par 2.
    """
    layers: list[nn.Module] = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


class GalaxyCNN(nn.Module):
    """CNN maison : 4 blocs conv + tête dense.

    channels=(32,64,128,256). AdaptiveAvgPool(4x4) fixe la tête quelle que soit
    la résolution d'entrée (64 ou 128 px). ~1,44 M paramètres, dominés par la
    première couche dense (4096 -> 256).

    use_bn et le dropout servent aux ablations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels=(32, 64, 128, 256),
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        use_bn: bool = True,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        prev = in_channels
        for c in channels:
            blocks.append(_conv_block(prev, c, use_bn=use_bn))
            prev = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flat = prev * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        # Nom de la dernière conv, utile pour Grad-CAM.
        self.last_conv = self.features[-1][0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_resnet18(pretrained: bool = True, mode: str = "finetune", num_classes: int = NUM_CLASSES):
    """ResNet18 avec dernière couche remplacée par `num_classes` sorties.

    mode='features'  : backbone gelé, on n'entraîne que fc (feature extraction).
    mode='finetune'  : tout est entraînable.
    """
    from torchvision import models

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    if mode == "features":
        for p in net.parameters():
            p.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
