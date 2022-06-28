from typing import Callable, Dict, Union

import torch
import torch.nn as nn
from ignite.engine import Engine
from torch.optim import Optimizer


def get_train_step_fn(classifier: nn.Module, criterion: nn.Module, optimizer: Optimizer,
                      device: Union[str, torch.device]) -> Callable:

    def _train_step_fn(engine: Engine, batch: Dict) -> float:
        classifier.train()
        img = batch['img'].to(device)
        target = batch['target'].to(device)
        pred = classifier(img)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return _train_step_fn


def get_eval_step_fn(classifier: nn.Module, device: Union[str, torch.device]):

    def _eval_step_fn(engine: Engine, batch: Dict) -> Dict:
        classifier.eval()
        with torch.no_grad():
            img = batch.pop('img')
            target = batch.pop('target')
            img = img.to(device)
            target = target.to(device)
            pred = classifier(img)

            batch.update({'pred': pred, 'target': target})
            return batch

    return _eval_step_fn
