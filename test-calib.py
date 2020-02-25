from typing import Tuple
import json
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data

from utils.arguments import Arguments


def init(model: nn.Module, device: torch.device,
         args: Arguments.parse.Namespace = None) \
        -> Tuple[nn.Module, None]:
    model = model.to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    return model, None


def test_calib(model: nn.Module, dataset: data.Dataset,
               device: torch.device = None, args: Arguments.parse.Namespace = None, **kwargs) \
        -> None:
    loader = data.DataLoader(dataset, args.batch, num_workers=args.worker,
                             shuffle=False, pin_memory=True)

    dest = Path(args.dest)
    dest.mkdir(exist_ok=True)
    corrects, total = 0, 0
    results = {}

    with tqdm(total=len(dataset)) as tq:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), tuple(map(lambda x: x.to(device), labels))

            with torch.set_grad_enabled(False):
                outputs = model(inputs)

            corrects += sum((output.argmax(dim=-1) == label.argmax(dim=-1)).sum()
                            for output, label in zip(outputs, labels)).item()
            total += len(labels) * args.batch

            tq.set_postfix(acc=corrects / total)
            tq.update(args.batch)

    results.update({
        'accuracy': float(corrects / total),
        'total': int(total),
    })

    yield results

    with open(str(dest.joinpath('results.json')), 'w') as f:
        json.dump(results, f)

    print(f'acc: {corrects / total}')
    print(f'total: {total}')
