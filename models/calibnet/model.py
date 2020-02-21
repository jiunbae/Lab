import torch
import torch.nn as nn
from torchvision import models

from models import Model


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class CalibNet(Model):
    def __init__(self, focal_size: int, distortion_size: int):
        super(CalibNet, self).__init__()

        self.focal_size = focal_size
        self.distortion_size = distortion_size

        self.feature = self.backbone()

        self.linear1 = nn.Linear(2048 + focal_size + distortion_size, 512)
        self.linear2 = nn.Linear(512, 1)

    @classmethod
    def new(cls, num_classes: int, focal: int, distortion: int, **kwargs):
        return cls(focal, distortion)

    @staticmethod
    def backbone() \
            -> nn.Module:
        feature = models.inception_v3(pretrained=True)
        feature.fc = nn.Identity(2048)

        return feature

    def forward(self, x: torch.Tensor, focal: torch.Tensor, distortion: torch.Tensor) \
            -> torch.Tensor:
        x = self.feature(x)

        return x
