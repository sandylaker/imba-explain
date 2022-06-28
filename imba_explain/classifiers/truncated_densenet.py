from collections import OrderedDict
from itertools import chain
from typing import Any

import timm
import torch
import torch.nn as nn
from timm.models.layers import BatchNormAct2d, create_classifier

from .builder import CUSTOM_CLASSIFIERS


@CUSTOM_CLASSIFIERS.register_module()
class TruncatedDenseNet(nn.Module):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        if 'model_name' in kwargs:
            raise ValueError(f"'model_name' is fixed to 'densenet121' and should not be specified in the arguments."
                             f"But got 'model_name': {kwargs['model_name']}.")
        deep_stem = 'deep' in kwargs.get('stem_type', '')
        densenet = timm.create_model(model_name='densenet121', **kwargs)
        ori_features = densenet.features

        if deep_stem:
            new_features = ['conv0', 'norm0', 'conv1', 'norm1', 'conv2', 'norm2', 'pool0']
        else:
            new_features = ['conv0', 'norm0', 'pool0']

        new_features.extend(['denseblock1', 'transition1', 'denseblock2'])
        new_features = OrderedDict([(name, ori_features.get_submodule(name)) for name in new_features])

        norm_layer = kwargs.get('norm_layer', BatchNormAct2d)
        new_features.update({'norm5': norm_layer(512)})

        self.features = nn.Sequential(new_features)
        self.num_classes = densenet.num_classes
        self.drop_rate = densenet.drop_rate
        self.global_pool, self.classifier = create_classifier(
            512, self.num_classes, pool_type=kwargs.get('global_pool', 'avg'))

        for m in chain(self.features.get_submodule('norm5').modules(), self.classifier.modules()):
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def get_classifier(self) -> nn.Module:
        return self.classifier

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.global_pool(x)

        x = self.classifier(x)
        return x
