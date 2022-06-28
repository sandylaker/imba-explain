from typing import Dict, Optional

import torch.nn as nn
from mmcv import Registry

LOSSES = Registry('losses')


def build_loss(cfg: Dict, default_args: Optional[Dict] = None) -> nn.Module:
    return LOSSES.build(cfg, default_args=default_args)
