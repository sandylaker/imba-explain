from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchray.attribution.extremal_perturbation import extremal_perturbation

from .builder import ATTRIBUTIONS, BaseAttribution


@ATTRIBUTIONS.register_module()
class ExtremalPerturbation(BaseAttribution):

    def __init__(self, attr_normalizer: Dict = dict(type='IdentityNormalizer'), **kwargs: Any) -> None:
        super(ExtremalPerturbation, self).__init__(attr_normalizer)

        self.extr_pert_kwargs = deepcopy(kwargs)
        self.extr_pert_fn: Optional[Callable] = None

    def set_classifier(self, classifier: nn.Module) -> None:

        def extr_pert_closure(img: Tensor, label: int) -> torch.Tensor:
            attr_map, _ = extremal_perturbation(classifier, input=img, target=label, **self.extr_pert_kwargs)
            return attr_map

        self.extr_pert_fn = extr_pert_closure

    def attribute(self, img: Tensor, label: Union[int, Tensor, np.ndarray], *args: Any, **kwargs: Any) -> np.ndarray:
        attr_map = self.extr_pert_fn(img, label)
        attr_map = attr_map.detach().cpu().numpy().mean((0, 1))
        return attr_map
