from typing import Iterator, Tuple, Union
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils import data
from torchvision.utils import save_image

from data import Dataset
from models import DataParallel
from utils.arguments import Arguments


def arguments(parser):
    parser.add_argument('--batch', required=False, default=32, type=int,
                        help="batch")
    parser.add_argument('--lr', required=False, default=.001, type=float,
                        help="learning rate")
    parser.add_argument('--b1', required=False, default=.5, type=float,
                        help="Beta1 for Adam optimizer")
    parser.add_argument('--b2', required=False, default=.999, type=float,
                        help="Beta2 for Adam optimizer")
    parser.add_argument('--parameters', required=False, nargs='+', default=[21, 22, 4, 4, 3, 3, 3, 3], type=int,
                        help='Parameters list size')
    parser.add_argument('--regression', required=False, action='store_true', default=False,
                        help='Regression options')

    parser.add_argument('--epoch', required=False, default=100000, type=int,
                        help="epoch")
    parser.add_argument('--start-epoch', required=False, default=0, type=int,
                        help="epoch start")
    parser.add_argument('--save-epoch', required=False, default=1000, type=int,
                        help="epoch for save")
    parser.add_argument('--worker', required=False, default=4, type=int,
                        help="worker")


def init(model: nn.Module, device: torch.device,
         args: Arguments.parse.Namespace = None) \
        -> Tuple[Union[DataParallel, nn.Module], object]:

    if args.model != 'None' and args.model != '':
        model.load(torch.load(args.model, map_location=lambda s, l: s))
    model.train()

    if device.type == 'cuda':
        model = DataParallel(model)
        model.state_dict = model.module.state_dict
        torch.backends.cudnn.benchmark = True
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    return model, optimizer


def train_calib(model: nn.Module, dataset: Dataset,
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device = None, args: Arguments.parse.Namespace = None, **kwargs) \
        -> Iterator[dict]:
    batch = args.batch

    loader = data.DataLoader(dataset, args.batch, num_workers=args.worker, drop_last=True,
                             shuffle=True, collate_fn=dataset.collate, pin_memory=True)
    iterator = iter(loader)

    with tqdm(total=args.epoch, initial=args.start_epoch) as tq:
        for iteration in range(args.start_epoch, args.epoch + 1):

            try:
                images, targets = next(iterator)

            except StopIteration:
                iterator = iter(loader)
                images, targets = next(iterator)

            inputs = Variable(images.to(device), requires_grad=False)
            targets = tuple(map(lambda x: Variable(x.to(device), requires_grad=False), targets))

            outputs = model(inputs)

            optimizer.zero_grad()

            if args.regression:
                loss = sum(criterion(output, target) for output, target in zip(outputs, targets))
            else:
                loss = sum(criterion(output, target) * np.log(output.size(1))
                           for output, target in zip(outputs, targets))

            loss.backward()
            optimizer.step()

            if args.save_epoch and not (iteration % args.save_epoch):
                torch.save(model.state_dict(),
                           str(Path(args.dest).joinpath(f'{args.name}-{iteration:06}.pth')))

                yield {
                    "iteration": iteration,
                    "loss": loss.item(),
                }

            tq.set_postfix(loss=loss.item())
            tq.update(1)
