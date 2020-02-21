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
    parser.add_argument('--latent', required=False, default=100, type=int,
                        help="Latent space dimensionality")
    parser.add_argument('--interval', required=False, default=400, type=int,
                        help="Image sampling interval")
    parser.add_argument('--size', required=False, default=256, type=int,
                        help="Size of image")

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
        -> Tuple[Union[nn.Module, Tuple], Union[nn.Module, Tuple]]:

    generator, discriminator = model
    #
    # if args.model != 'None' and args.model != '':
    #     model.load(torch.load(args.model, map_location=lambda s, l: s))
    generator.train()
    discriminator.train()

    # if device.type == 'cuda':
    #     model = DataParallel(model)
    #     model.state_dict = model.module.state_dict
    #     torch.backends.cudnn.benchmark = True
    generator.to(device)
    discriminator.to(device)

    generator_optim = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    return (generator, discriminator), (generator_optim, discriminator_optim)


def train_gan(model: nn.Module, dataset: Dataset,
              criterion: nn.Module, optimizer: optim.Optimizer,
              device: torch.device = None, args: Arguments.parse.Namespace = None, **kwargs) \
        -> Iterator[dict]:
    batch = args.batch

    loader = data.DataLoader(dataset, args.batch, num_workers=args.worker, drop_last=True,
                             shuffle=True, collate_fn=Dataset.collate, pin_memory=True)
    iterator = iter(loader)
    generator, discriminator = model
    generator_optim, discriminator_optim = optimizer

    with tqdm(total=args.epoch, initial=args.start_epoch) as tq:
        for iteration in range(args.start_epoch, args.epoch + 1):

            try:
                images, targets = next(iterator)

            except StopIteration:
                iterator = iter(loader)
                images, targets = next(iterator)

            images = Variable(images.to(device), requires_grad=False)
            # targets = [Variable(target.to(device), requires_grad=False) for target in targets]
            targets = Variable(torch.LongTensor(batch).to(device).fill_(0), requires_grad=False)

            input_valid = Variable(torch.FloatTensor(batch, 1).to(device).fill_(1.), requires_grad=False)
            input_fake = Variable(torch.FloatTensor(batch, 1).to(device).fill_(0.), requires_grad=False)

            # == Train Generator ==
            generator_optim.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch, args.latent))).to(device))
            gen_labels = Variable(torch.LongTensor(np.random.randint(0, args.classes, batch)).to(device))

            # Generate a batch of images
            input_generated = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(input_generated, gen_labels)
            g_loss = criterion(validity, input_valid)

            g_loss.backward()
            generator_optim.step()

            # == Train Discriminator ==
            discriminator_optim.zero_grad()

            # Loss for real images
            validity_real = discriminator(images, targets)
            d_real_loss = criterion(validity_real, input_valid)

            # Loss for fake images
            validity_fake = discriminator(input_generated.detach(), gen_labels)
            d_fake_loss = criterion(validity_fake, input_fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            discriminator_optim.step()

            if torch.isnan(g_loss) and torch.isnan(d_loss):
                print(f'NaN detected in {iteration}')

            tq.set_postfix(g_loss=g_loss.item(), d_loss=d_loss.item())
            tq.update(1)

            if args.save_epoch and not (iteration % args.save_epoch):
                Path('images').mkdir(exist_ok=True)
                grid = 3

                z = Variable(torch.FloatTensor(np.random.normal(0, 1, (grid ** 2, args.latent))).to(device))
                # Get labels ranging from 0 to n_classes for n rows
                labels = np.array([num for _ in range(grid) for num in range(grid)])
                labels = Variable(torch.LongTensor(labels).to(device).fill_(0))
                save_image(generator(z, labels).data, f"images/{iteration:02}.png", nrow=grid, normalize=True)

                # torch.save(model.state_dict(),
                #            str(Path(args.dest).joinpath(f'{args.name}-{iteration:06}.pth')))

                yield {
                    "iteration": iteration,
                    "g_loss": g_loss.item(),
                    "d_loss": d_loss.item(),
                }
