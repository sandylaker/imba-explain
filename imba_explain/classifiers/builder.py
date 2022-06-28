from copy import deepcopy

import timm
import torch
import torch.nn as nn
from mmcv import Registry
from torchvision import models

__all__ = ['build_classifier', 'TIMM_CLASSIFIERS', 'TORCHVISION_CLASSIFIERS', 'CUSTOM_CLASSIFIERS', 'get_module']

from typing import Dict, Optional, Union


def _preprocess_cfg(cfg: Dict, default_args: Optional[Dict] = None) -> Dict:
    cfg = deepcopy(cfg)
    if default_args is not None:
        for name, value in default_args.items():
            cfg.setdefault(name, value)
    return cfg


def build_timm_classifier(registry: Registry, cfg: Dict, default_args: Optional[Dict] = None) -> nn.Module:
    cfg = _preprocess_cfg(cfg, default_args=default_args)
    model_name = cfg.pop('type')
    return timm.create_model(model_name, **cfg)


def build_torchvision_classifier(registry: Registry, cfg: Dict, default_args: Optional[Dict] = None) -> nn.Module:
    cfg = _preprocess_cfg(cfg, default_args=default_args)
    model_name = cfg.pop('type')
    ckpt_path = cfg.pop('checkpoint_path', None)
    _builder = getattr(models, model_name)
    model = _builder(**cfg)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
    return model


TIMM_CLASSIFIERS = Registry('timm_classifiers', scope='timm', build_func=build_timm_classifier)
TORCHVISION_CLASSIFIERS = Registry(
    'torchvision_classifiers', scope='torchvision', build_func=build_torchvision_classifier)
CUSTOM_CLASSIFIERS = Registry('custom_classifiers', scope='custom')


def build_classifier(cfg: Dict, default_args: Optional[Dict] = None) -> nn.Module:
    cfg = deepcopy(cfg)
    if 'type' not in cfg:
        raise ValueError("Key 'type' must be contained in the config.")
    # timm and torchvision registries are actually empty. If using default build_from_cfg function to build
    # from CLASSIFIERS, it will raise an KeyError. We need to find the scope and call the corresponding build function
    scope, model_name = Registry.split_scope_key(cfg['type'])
    if scope is None:
        raise ValueError(f"type must be in format <scope>.<model_name>, but got {cfg['type']}.")
    if scope == '':
        raise ValueError('scope must not be an empty string.')
    cfg.update({'type': model_name})

    if scope == 'timm':
        return TIMM_CLASSIFIERS.build(cfg=cfg, default_args=default_args)
    elif scope == 'torchvision':
        return TORCHVISION_CLASSIFIERS.build(cfg=cfg, default_args=default_args)
    elif scope == 'custom':
        return CUSTOM_CLASSIFIERS.build(cfg=cfg, default_args=default_args)
    else:
        raise ValueError(f"Invalid scope name, should be one of 'timm', 'torchvision', 'custom', but got {scope}.")


def get_module(model: nn.Module, module: Union[str, nn.Module]) -> Optional[nn.Module]:
    r"""Returns a specific layer in a model based.
    Shameless copy from `<TorchRay https://github.com/facebookresearch/TorchRay>_`.
    :attr:`module` is either the name of a module (as given by the
    :func:`named_modules` function for :class:`torch.nn.Module` objects) or
    a :class:`torch.nn.Module` object. If :attr:`module` is a
    :class:`torch.nn.Module` object, then :attr:`module` is returned unchanged.
    If :attr:`module` is a str, the function searches for a module with the
    name :attr:`module` and returns a :class:`torch.nn.Module` if found;
    otherwise, ``None`` is returned.
    Args:
        model: Model in which to search for layer.
        module: Name of layer (str) or the layer itself (:class:`torch.nn.Module`).
    Returns:
        Specific PyTorch layer (``None`` if the layer isn't found).
    """
    if not isinstance(module, (str, torch.nn.Module)):
        raise TypeError(f'module can only be a str or torch.nn.Module, but got {module.__class__.__name__}')
    if isinstance(module, torch.nn.Module):
        return module

    if module == '':
        return model

    for name, curr_module in model.named_modules():
        if name == module:
            return curr_module

    return None
