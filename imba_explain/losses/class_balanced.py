from typing import Optional

import torch.nn as nn
from ignite.utils import setup_logger
from torch import Tensor

from .builder import LOSSES
from .focal import sigmoid_focal_loss


@LOSSES.register_module()
class ClassBalancedLoss(nn.Module):
    r""" Class-balanced loss. See `paper <https://arxiv.org/abs/1901.05555>`__ for details.

    Args:
        beta: Hyperparameter for computing the effective number of samples.
        loss_weight: Weight of loss.
        gamma: Focusing parameter in focal loss
        reduction: The method used to reduce the loss into a scalar.

    """

    def __init__(self,
                 beta: float = 0.9999,
                 loss_weight: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'mean') -> None:
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.reduction = reduction

        self.register_buffer('alpha', None)

    def receive_data_dist_info(self, num_pos_neg: Tensor) -> None:
        shape = num_pos_neg.shape
        if not (len(shape) == 2 and shape[0] == 2):
            raise ValueError(f'num_pos_neg should have shape (2, num_classes), but got {shape}.')
        num_pos, num_neg = num_pos_neg[0], num_pos_neg[1]
        alpha = (1 - self.beta**num_neg) / (1 - self.beta**num_pos)
        alpha /= (1 + alpha)
        logger = setup_logger('imba-explain')
        logger.info(f'Alpha of class-imbalanced focal loss: {alpha.numpy()}')
        self.alpha = alpha

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * sigmoid_focal_loss(
            pred, target, weight, gamma=self.gamma, alpha=self.alpha, reduction=reduction, avg_factor=avg_factor)
        return loss_cls
