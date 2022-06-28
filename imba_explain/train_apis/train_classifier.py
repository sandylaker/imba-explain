import os.path as osp
from copy import deepcopy
from datetime import datetime

import ignite.distributed as idist
import mmcv
from functools import partial
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC, AveragePrecision
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine, param_scheduler
from ignite.metrics import Accuracy
from ignite.utils import manual_seed, setup_logger
from mmcv import Config
from mmcv.runner import build_optimizer
from torch.utils.data import WeightedRandomSampler

from ..classifiers import build_classifier
from ..datasets import build_dataset
from ..losses import build_loss
from .eval_hooks import MetricsTextLogger
from .step_fn import get_eval_step_fn, get_train_step_fn
from .train_hooks import TrainStatsTextLogger
from .utils import logits_transform, prob_transform


def train_classifier(local_rank: int, cfg: Config) -> None:
    rank = idist.get_rank()
    manual_seed(cfg.get('seed', 2022) + rank)
    device = idist.device()

    logger = setup_logger(
        'imba-explain', filepath=osp.join(cfg.work_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    env_info_dict = mmcv.collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    logger.info(f'Config:\n{cfg.pretty_text}')

    train_set = build_dataset(cfg.data['train'])
    val_set = build_dataset(cfg.data['val'])
    logger.info(f'Training set size: {len(train_set)} samples. Validation set size: {len(val_set)} samples.')

    data_loader_cfg = deepcopy(cfg.data['data_loader'])
    use_weighted_sampler = data_loader_cfg.pop('weighted_sampler', False)
    if use_weighted_sampler:
        if not hasattr(train_set, 'imba_sampling_weights'):
            raise ValueError('The training  dataset class must implement the method imba_sampling_weights, '
                             'when weighted_sampler in data_loader config is True.')
        weights = train_set.imba_sampling_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(train_set))
        data_loader_cfg.update({'sampler': sampler, 'shuffle': False})
    train_loader = idist.auto_dataloader(train_set, **data_loader_cfg)
    data_loader_cfg.update({'shuffle': False, 'sampler': None})
    val_loader = idist.auto_dataloader(val_set, **data_loader_cfg)
    epoch_length = len(train_loader)

    classifier = build_classifier(cfg.classifier)
    classifier.to(device)
    classifier = idist.auto_model(classifier, sync_bn=cfg.get('sync_bn', False))

    # build trainer
    optimizer = build_optimizer(classifier, cfg.optimizer)
    criterion = build_loss(cfg.loss)
    # let the loss function receive the data distribution information
    if hasattr(criterion, 'receive_data_dist_info'):
        criterion.receive_data_dist_info(train_set.get_num_pos_neg())
    criterion.to(device)

    try:
        has_parameter = next(criterion.parameters()) is not None
    except StopIteration:
        has_parameter = False

    if has_parameter:
        # in case where loss function contains learnable parameters, use DDP
        criterion = idist.auto_model(criterion)
        # when the loss function has learnable parameters, add them to the optimizer's parameter group
        optimizer.add_param_group({'params': criterion.parameters()})
    optimizer = idist.auto_optim(optimizer)
    trainer = Engine(get_train_step_fn(classifier, criterion, optimizer, device))
    trainer.logger = logger

    # built evaluator
    eval_step_fn = get_eval_step_fn(classifier, device)
    evaluator = Engine(eval_step_fn)
    evaluator.logger = logger

    # evaluator handlers
    pbar = ProgressBar(persist=True)
    pbar.attach(evaluator)

    # pred_select_inds is only used in cross-dataset test
    _prob_transform = partial(prob_transform, pred_select_inds=None, target_select_inds=None)
    _logits_transform = partial(logits_transform, pred_select_inds=None, target_select_inds=None)

    val_metrics = {
        'accuracy': Accuracy(output_transform=_prob_transform, device=device, **cfg.class_metrics['accuracy']),
        'roc_auc': ROC_AUC(output_transform=_logits_transform, device=device, **cfg.class_metrics['roc_auc']),
        'ap': AveragePrecision(output_transform=_logits_transform, device=device, **cfg.class_metrics['ap'])
    }
    for name, metric in val_metrics.items():
        metric.attach(engine=evaluator, name=name)

    metrics_logger = MetricsTextLogger(logger=logger)
    metrics_logger.attach(evaluator, trainer)

    # trainer handlers
    def run_validation(engine_train: Engine, engine_val: Engine) -> None:
        engine_val.run(val_loader)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=cfg.val_interval), run_validation, evaluator)

    if cfg.cosine_annealing:
        cycle_size = cfg.max_epochs * epoch_length
        lr = cfg.optimizer['lr']
        lr_scheduler = param_scheduler.CosineAnnealingScheduler(
            optimizer=optimizer, param_name='lr', start_value=lr, end_value=lr * 0.01, cycle_size=cycle_size)
        lr_scheduler = param_scheduler.create_lr_scheduler_with_warmup(
            lr_scheduler, warmup_start_value=0.01 * lr, warmup_duration=1000, warmup_end_value=lr)
        trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    to_save = {'classifier': classifier}
    save_handler = DiskSaver(osp.join(osp.join(cfg.work_dir, 'ckpts')), require_empty=False)
    score_fn = Checkpoint.get_default_score_fn('roc_auc')
    ckpt_handler = Checkpoint(
        to_save,
        save_handler,
        n_saved=cfg.get('n_saved', 1),
        score_name='roc_auc',
        score_function=score_fn,
        global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
        greater_or_equal=True)
    evaluator.add_event_handler(Events.COMPLETED, ckpt_handler)

    train_stats_logger = TrainStatsTextLogger(interval=cfg.log_interval, logger=logger)
    train_stats_logger.attach(trainer, optimizer)

    trainer.run(data=train_loader, max_epochs=cfg.max_epochs)
