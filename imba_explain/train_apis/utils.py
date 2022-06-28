from typing import Dict, List, Optional, Tuple, Union

from torch import Tensor


def prob_transform(batch: Dict[str, Union[Tensor, str]],
                   pred_select_inds: Optional[List[int]] = None,
                   target_select_inds: Optional[List[int]] = None) -> Tuple[Tensor, Tensor]:
    pred = batch['pred'].sigmoid()
    target = batch['target']
    if pred_select_inds is not None:
        pred = pred[:, pred_select_inds]
    if target_select_inds is not None:
        target = target[:, target_select_inds]

    return (pred >= 0.5).to(target), target


def logits_transform(batch: Dict[str, Union[Tensor, str]],
                     pred_select_inds: Optional[List[int]] = None,
                     target_select_inds: Optional[List[int]] = None) -> Tuple[Tensor, Tensor]:
    pred = batch['pred']
    target = batch['target']
    if pred_select_inds is not None:
        pred = pred[:, pred_select_inds]
    if target_select_inds is not None:
        target = target[:, target_select_inds]
    return pred, target
