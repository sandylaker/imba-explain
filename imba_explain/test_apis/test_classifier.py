import os.path as osp
from copy import deepcopy
from typing import Union

import ignite.distributed as idist
import mmcv
import torch
from functools import partial
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC, AveragePrecision
from ignite.engine import Engine
from ignite.metrics import Accuracy
from ignite.utils import manual_seed, setup_logger
from mmcv import Config

from ..classifiers import build_classifier
from ..datasets import build_dataset
from ..train_apis import MetricsTextLogger, PredictionsSaver, get_eval_step_fn, logits_transform, prob_transform


def test_classifier(cfg: Config, ckpt: str, device: Union[str, torch.device] = 'cuda:0') -> None:
    """Test a classifier. This function only support the non-distributed setting.

    Args:
        cfg: config file.

    Returns:
        None
    """
    manual_seed(cfg.get('seed', 2022))

    logger = setup_logger('imba-explain')

    test_set = build_dataset(cfg.data['test'])  # noqa
    if not hasattr(test_set, 'class_names'):
        raise ValueError('Dataset class should have attribute class_name.')
    data_loader_cfg = deepcopy(cfg.data['data_loader'])
    data_loader_cfg.update({'shuffle': False})
    test_loader = idist.auto_dataloader(test_set, **data_loader_cfg)

    state_dict = torch.load(ckpt, map_location='cpu')
    logger.info(f'Using the checkpoint: {ckpt}')
    classifier = build_classifier(cfg.classifier)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    eval_step_fn = get_eval_step_fn(classifier, device)
    evaluator = Engine(eval_step_fn)
    evaluator.logger = logger

    pbar = ProgressBar(persist=True)
    pbar.attach(evaluator)

    target_select_names = test_set.select_class_names
    target_select_inds = test_set.select_class_inds
    pred_select_inds = cfg.get('pred_select_inds', None)
    _prob_transform = partial(prob_transform, pred_select_inds=pred_select_inds, target_select_inds=target_select_inds)
    _logits_transform = partial(
        logits_transform, pred_select_inds=pred_select_inds, target_select_inds=target_select_inds)

    test_metrics = {
        'accuracy': Accuracy(output_transform=_prob_transform, device=device, **cfg.class_metrics['accuracy']),
        'roc_auc': ROC_AUC(output_transform=_logits_transform, device=device, **cfg.class_metrics['roc_auc']),
        'ap': AveragePrecision(output_transform=_logits_transform, device=device, **cfg.class_metrics['ap'])
    }
    for name, metric in test_metrics.items():
        metric.attach(engine=evaluator, name=name)

    metrics_logger = MetricsTextLogger(logger=logger)
    metrics_logger.attach(evaluator)

    test_results_dir = osp.join(cfg.work_dir, 'test_results')
    mmcv.mkdir_or_exist(test_results_dir)
    preds_fp = osp.join(test_results_dir, 'predictions.csv')

    preds_saver = PredictionsSaver(preds_fp, target_select_names, target_select_inds, pred_select_inds, logger=logger)
    preds_saver.attach(evaluator)

    evaluator.run(data=test_loader)
