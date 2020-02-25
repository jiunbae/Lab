import torch
import torch.optim as optim

from models import Model
from data.dataset import Dataset
from utils import seed
from utils.executable import Executable
from utils.arguments import Arguments
from utils.augmentation import Augmentation


def main(args: Arguments.parse.Namespace):
    executor = Executable.s[args.command]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Augmentation.get(args.type)(size=(args.size, args.size))
    dataset = Dataset.get(args.type)(args.dataset,
                                     transform=transform, train=args.command == 'train', **vars(args))

    num_classes = args.classes or dataset.num_classes
    args.classes = num_classes

    model = Model.get(args.backbone).new(num_classes, **vars(args))
    Executable.log('Model', model)

    model, optimizer = executor.init(model, device, args)

    criterion = model.loss(num_classes, device=device)

    executor(model, dataset=dataset, criterion=criterion, optimizer=optimizer,
             device=device, args=args)

    Executable.close()


if __name__ == '__main__':
    arguments = Arguments()

    seed(arguments.seed)
    main(arguments)
