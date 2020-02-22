from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models import Model


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class swish(nn.Module):
    def __init__(self, inplace=True):
        super(swish, self).__init__()
        self.activation = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * (self.relu(x + 3) / 6)


class CalibNet(Model):
    LOSS = LogCoshLoss

    def __init__(self, focal_x_size: int, focal_y_size: int, batch_size: int):
        super(CalibNet, self).__init__()
        self.batch_size_ = batch_size
        self.focal_x_size = focal_x_size
        self.focal_y_size = focal_y_size

        self.feature = self.backbone()

        self.focal_x = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.focal_y = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    @classmethod
    def new(cls, num_classes: int, focal: int, distortion: int, batch: int, **kwargs):
        return cls(focal, distortion, batch)

    @staticmethod
    def backbone() \
            -> nn.Module:
        feature = models.resnet101(pretrained=True)
        feature.fc = nn.Identity(2048)

        return feature

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.feature(x)

        focal_x = self.focal_x(feature)
        focal_y = self.focal_y(feature)

        return focal_x, focal_y
