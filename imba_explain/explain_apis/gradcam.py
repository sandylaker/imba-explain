from typing import Any, Dict, Optional, Union

import numpy as np
import torch.nn as nn
from captum.attr import LayerGradCam
from torch import Tensor

from ..classifiers import get_module
from .builder import ATTRIBUTIONS, BaseAttribution


@ATTRIBUTIONS.register_module()
class GradCAM(BaseAttribution):

    def __init__(self,
                 target_layer: str,
                 attribute_to_layer_input: bool = False,
                 relu_attributions: bool = True,
                 attr_normalizer: Dict = dict(type='IdentityNormalizer')) -> None:
        super(GradCAM, self).__init__(attr_normalizer)
        self.target_layer = target_layer
        self.attribute_to_layer_input = attribute_to_layer_input
        self.relu_attributions = relu_attributions

        self.grad_cam: Optional[LayerGradCam] = None

    def set_classifier(self, classifier: nn.Module) -> None:
        target_layer = get_module(classifier, self.target_layer)
        self.grad_cam = LayerGradCam(classifier, target_layer)

    def attribute(self, img: Tensor, label: Union[int, Tensor, np.ndarray], *args: Any, **kwargs: Any) -> np.ndarray:
        if isinstance(label, (Tensor, np.ndarray)):
            raise TypeError(f'For now GradCAM only supports integer label, but got {label.__class__.__name__}')
        attr_map = self.grad_cam.attribute(
            img,
            int(label),
            relu_attributions=self.relu_attributions,
            attribute_to_layer_input=self.attribute_to_layer_input)

        if attr_map.shape[:2] != (1, 1):
            raise ValueError(f'Attribution map has incorrect shape: {attr_map.shape}. '
                             f'A valid shape should be (1, 1, height, width).')

        attr_map = attr_map.squeeze(0).squeeze(0).detach().cpu().numpy()
        return attr_map
