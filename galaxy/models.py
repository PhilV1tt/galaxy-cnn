"""Project models: convolution-free baselines, custom CNN, ResNet18.

The project's story is a three-level ladder:
  1. baselines (logistic regression, MLP) blind to spatial structure;
  2. custom CNN (convolution = weight sharing + translation equivariance);
  3. transfer learning (ImageNet-pretrained ResNet18).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import NUM_CLASSES


class LogisticRegression(nn.Module):
    """Baseline 1: multinomial logistic regression on flattened pixels.

    A single linear layer. No notion of locality: each pixel is an
    independent variable. Deliberately mediocre.
    """

    def __init__(self, in_channels: int = 3, image_size: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_features = in_channels * image_size * image_size
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class MLP(nn.Module):
    """Baseline 2: multilayer perceptron on flattened pixels.

    Two hidden layers + ReLU + dropout. More expressive than logistic
    regression, but still blind to 2D structure.
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
    """Block [Conv 3x3 pad 1 -> (BatchNorm) -> ReLU -> MaxPool 2].

    The 3x3 pad 1 conv keeps the spatial size; pooling halves it.
    """
    layers: list[nn.Module] = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


class GalaxyCNN(nn.Module):
    """Custom CNN: 4 conv blocks + dense head.

    channels=(32,64,128,256). AdaptiveAvgPool(4x4) fixes the head whatever
    the input resolution (64 or 128 px). ~1,44 M parameters, dominated by the
    first dense layer (4096 -> 256).

    use_bn and dropout are used for the ablations.
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
        # Name of the last conv, useful for Grad-CAM.
        self.last_conv = self.features[-1][0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_resnet18(pretrained: bool = True, mode: str = "finetune", num_classes: int = NUM_CLASSES):
    """ResNet18 with the last layer replaced by `num_classes` outputs.

    mode='features'  : frozen backbone, only fc is trained (feature extraction).
    mode='finetune'  : everything is trainable.
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
