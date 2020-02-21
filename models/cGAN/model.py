from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from models import Model


class Generator(nn.Module):
    def __init__(self, num_classes: int, shape: Tuple[int, int, int], latent: int):
        super(Generator, self).__init__()

        self.shape = shape
        self.label_emb = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), * self.shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes: int, shape: Tuple[int, int, int]):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class CGAN(Model):
    LOSS = torch.nn.MSELoss

    @classmethod
    def new(cls, num_classes: int, size: int, latent: int,
            *args, **kwargs):
        shape = (3, size, size)

        generator = Generator(num_classes, shape, latent)
        discriminator = Discriminator(num_classes, shape)

        return generator, discriminator
