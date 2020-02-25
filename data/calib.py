from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import skimage
import torch

from data import Dataset


class Calib(Dataset):
    class_id = 1

    IMAGE_DIR = 'images'
    IMAGE_EXT = '.jpg'
    CALIB_DIR = 'calibration'
    CALIB_EXT = '.txt'

    PARAMETER = {
        'center_x': np.arange(1860, 2070, 10),
        'center_y': np.arange(1325, 1540, 10),
        'focal_x': np.array([980, 1400, 1514, 1540]),
        'focal_y': np.array([980, 1400, 1514, 1520]),
        'up_0': np.array([-1, 0, 1]), 'up_1': np.array([-1, 0, 1]),
        'right_0': np.array([-1, 0, 1]), 'right_1': np.array([-1, 0, 1]),
    }
    PARAMETER_INDEX = {
        'center_x': (0, 0), 'center_y': (0, 1),
        'focal_x': (0, 2), 'focal_y': (0, 3),
        'up_0': (2, 0), 'up_1': (2, 1),
        'right_0': (3, 0), 'right_1': (3, 1),
    }

    SHAPE = 300, 300

    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 train: bool = True,
                 eval_only: bool = False,
                 regression: bool = False,
                 **kwargs):
        self.name = 'Calibration'

        path, *options = root.split(':')

        self.root = Path(path)
        self.transform = transform
        self.target_transform = target_transform or None
        self.eval_only = eval_only
        self.regression = regression

        # Update options
        for option in options:
            key, value = map(str.strip, option.split('='))
            setattr(self, key, int(value))

        if eval_only:
            self.images = list(sorted(self.root.glob(f'*{self.IMAGE_EXT}')))

        else:
            self.images = list(sorted(self.root.joinpath(self.IMAGE_DIR).glob(f'*{self.IMAGE_EXT}')))
            self.calibrations = self.root.joinpath(self.CALIB_DIR)

        self.shape = self.pull_image(0).shape

    @staticmethod
    def collate(batch):
        images, targets, *_ = zip(*batch)
        targets = tuple(map(torch.Tensor, zip(*targets)))

        return torch.stack(images, dim=0), targets

    def __getitem__(self, index):
        item = self.pull_item(index)

        return item

    def __len__(self):
        return len(self.images)

    def pull_name(self, index: int):
        return self.images[index].stem

    def pull_item(self, index: int):
        image = self.pull_image(index)
        height, width, channels = image.shape

        if self.eval_only is None:
            uniques = np.arange(0)
            boxes = np.empty((uniques.size, 4))
            labels = np.empty((uniques.size, 1))

        else:
            targets = self.pull_anno(index)

            if self.target_transform is not None:
                targets = self.target_transform(targets)

        if self.transform is not None:
            image, targets = self.transform(image, targets)

        return torch.from_numpy(image).permute(2, 0, 1), targets

    def pull_image(self, index: int) \
            -> np.ndarray:

        image = skimage.io.imread(str(self.images[index]))

        return image

    def pull_anno(self, index: int) \
            -> Tuple[np.ndarray, ...]:
        def one_hot(source: np.ndarray, target: np.ndarray) \
                -> np.ndarray:
            index = np.absolute(np.expand_dims(source, axis=1) - target).argmin(axis=0).squeeze()

            return np.eye(source.size, dtype=np.int)[index]

        try:
            name = self.pull_name(index)
            calibration = next(self.calibrations.glob(f"{'_'.join(name.split('_')[:2])}{self.CALIB_EXT}"))

            calib = pd.read_csv(str(calibration), sep=' ', header=None).values

        except (pd.errors.EmptyDataError, IndexError):
            calib = np.zeros((3, 4), dtype=np.float32)

        return tuple(
            (calib[self.PARAMETER_INDEX[key]]
             if self.regression else one_hot(param, calib[self.PARAMETER_INDEX[key]]))
            for key, param in self.PARAMETER.items()
        )
