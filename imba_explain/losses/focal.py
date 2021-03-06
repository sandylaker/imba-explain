# Copied from https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/losses/focal_loss.py
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .builder import LOSSES
from .utils import weight_reduce_loss


def sigmoid_focal_loss(pred: Tensor,
                       target: Tensor,
                       weight: Optional[Tensor] = None,
                       gamma: float = 2.0,
                       alpha: Union[float, Tensor] = 0.25,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    r"""Sigmoid focal loss.
    Args:
        pred: The prediction with shape (N, \*).
        target: The ground truth label of the prediction with
            shape (N, \*).
        weight: Sample-wise loss weight with shape
            (N, ). Defaults to None.
        gamma: The gamma for calculating the modulating factor.
            Defaults to 2.0.
        alpha: A balanced form for Focal Loss. If it is a float, then a global balanced form is applied.
            If it is Tensor with shape (N, \*) or any shape that are broadcast-compatible with `pred`.
        reduction: The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' ,
            loss is same shape as pred and label. Defaults to 'mean'.
        avg_factor: Average factor that is used to average
            the loss. Defaults to None.
    Returns:
        Loss.
    """
    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):
    """Focal loss.

    Args:
        gamma (float): Focusing parameter in focal loss.
            Defaults to 2.0.
        alpha (float): The parameter in balanced form of focal
            loss. Defaults to 0.25.
        reduction (str): The method used to reduce the loss into
            a scalar. Options are "none" and "mean". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        r"""Sigmoid focal loss.
        Args:
            pred (torch.Tensor): The prediction with shape (N, \*).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, \*), N or (N,1). Note that the target must be one-hot encoded
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \*). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.
        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * sigmoid_focal_loss(
            pred, target, weight, gamma=self.gamma, alpha=self.alpha, reduction=reduction, avg_factor=avg_factor)
        return loss_cls
