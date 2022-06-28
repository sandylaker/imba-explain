from typing import Dict, List, Optional, Union

import albumentations as A
import inspect
from albumentations.pytorch.transforms import ToTensorV2
from mmcv import Registry
from torch.utils.data import Dataset

DATASETS = Registry('datasets')
PIPELINES = Registry('pipelines')


def register_albu_transforms() -> List:
    albu_transforms = []
    for module_name in dir(A):
        if module_name.startswith('_'):
            continue
        transform = getattr(A, module_name)
        if inspect.isclass(transform) and issubclass(transform, A.BasicTransform):
            PIPELINES.register_module()(transform)
            albu_transforms.append(module_name)
    return albu_transforms


albu_transforms = register_albu_transforms()
PIPELINES.register_module(module=ToTensorV2)


def build_pipeline(cfg: Union[Dict, List], default_args: Optional[Dict] = None) -> Union[A.BasicTransform, A.Compose]:
    if isinstance(cfg, Dict):
        return PIPELINES.build(cfg)
    else:
        pipeline = []
        for transform_cfg in cfg:
            t = build_pipeline(transform_cfg)
            pipeline.append(t)
        if default_args is None:
            default_args = {}
        return A.Compose(pipeline, **default_args)


def build_dataset(cfg: Dict, default_args: Optional[Dict] = None) -> Dataset:
    return DATASETS.build(cfg, default_args=default_args)
