from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .builder import LOSSES
from .utils import weight_reduce_loss


def multi_class_loss(pred: Tensor,
                     label: Tensor,
                     log_variance: torch.Tensor,
                     weight: Optional[Tensor] = None,
                     pos_weight: Optional[Tensor] = None,
                     reduction: str = 'mean',
                     avg_factor: Optional[float] = None) -> Tensor:
    assert pred.dim() == label.dim()
    assert log_variance.dim() == 1

    # inter class weight, exp(- log_variance) = 1 / sigma^2. Please see the formulas in the original paper.
    class_weight = torch.exp(-log_variance)
    class_weight = class_weight.repeat(pred.size()[0], 1)

    # manually multiply class_weight with raw bce loss,
    # since the bce loss does not support back propagating through class_weight
    loss = F.binary_cross_entropy_with_logits(pred, label, pos_weight=pos_weight, reduction='none')
    loss = class_weight * loss

    # apply sample-wise weight
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)

    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    loss += 0.5 * torch.sum(log_variance)
    return loss


@LOSSES.register_module()
class MultiTaskLoss(nn.Module):
    r"""Multi-task loss. See `paper <https://arxiv.org/abs/1705.07115>`__ for details.

    Args:
        num_classes: Number of classes.
        loss_weight: weight factor multiplied with the multi-task loss.
        reduction: The method to reduce the loss into a scalar.
        device: device to allocate the parameters.

    """

    def __init__(self,
                 num_classes: int = 15,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 device: Optional[Union[str, torch.device]] = None) -> None:
        super(MultiTaskLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.log_variance = nn.Parameter(torch.zeros(self.num_classes, device=device))

    def forward(self,
                cls_scores: Tensor,
                label: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        loss_cls = multi_class_loss(
            cls_scores, label, self.log_variance, weight=weight, reduction=reduction, avg_factor=avg_factor)

        loss_cls = self.loss_weight * loss_cls
        return loss_cls
