from collections import defaultdict
from itertools import chain
from logging import Logger
from typing import List, Optional

import ignite.distributed as idist
import numpy as np
import pandas as pd
import torch
from ignite.engine import Engine, Events
from ignite.utils import setup_logger
from tabulate import tabulate


class MetricsTextLogger:

    def __init__(self, logger: Optional[Logger] = None) -> None:
        self.logger = logger if logger is not None else setup_logger('imba-explain')

    def _log_metrics(self, evaluator: Engine, trainer: Optional[Engine] = None) -> None:
        if trainer is not None:
            epoch = trainer.state.epoch
            max_epochs = trainer.state.max_epochs
            log_str = f'Epoch [{epoch}/{max_epochs}]: '
        else:
            log_str = 'Evaluation metrics: '
        log_str += '; '.join([f'{name}: {val:.4f}' for name, val in evaluator.state.metrics.items()])
        self.logger.info(log_str)

    def attach(self, evaluator: Engine, trainer: Optional[Engine] = None) -> None:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, self._log_metrics, trainer)


class PredictionsSaver:

    def __init__(self,
                 file_path: str,
                 target_select_names: List[str],
                 target_select_inds: List[int],
                 pred_select_inds: Optional[List[int]] = None,
                 logger: Optional[Logger] = None) -> None:
        if idist.get_world_size() > 1:
            raise RuntimeError(f'PredictionSaver only support the non-distributed setting, '
                               f'but the world size is {idist.get_world_size()}.')
        if not file_path.endswith('.csv'):
            raise ValueError(f"file path must be a string ending with '.csv', but got {file_path}.")
        pred_select_inds = list(range(len(target_select_names))) if pred_select_inds is None \
            else pred_select_inds
        if len(target_select_names) != len(pred_select_inds) or len(target_select_names) != len(target_select_inds):
            raise ValueError(f'data_select_class_names, data_select_inds, and pred_select_inds '
                             f'should have equal length as pred_select_class_inds, '
                             f'but got {target_select_names}, and {target_select_inds}, and {pred_select_inds}.')
        self.target_select_names = target_select_names
        self.target_select_inds = target_select_inds
        self.pred_select_inds = pred_select_inds
        self.file_path = file_path
        self.logger = logger if logger is not None else setup_logger('imba-explain')
        tabular_data = {
            'Target Class Names': self.target_select_names,
            'Target Class Indices': self.target_select_inds,
            'Prediction Class Indices': self.pred_select_inds
        }
        table = tabulate(tabular_data, headers='keys', tablefmt='pretty', stralign='left', numalign='left')
        log_str = 'Prediction Saver: \n'
        log_str += f'{table}'
        self.logger.info(log_str)

        self.buffers = defaultdict(list)
        self._has_saved = False

    def _init_buffers(self, engine: Engine) -> None:
        self.buffers.clear()

    def _update_buffers(self, engine: Engine) -> None:
        output = engine.state.output
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                if key == 'pred':
                    value = value.sigmoid()
                value = value.detach().cpu().numpy()
            self.buffers[key].append(value)

    def _save_buffers(self, engine: Engine) -> None:
        if not self._has_saved:
            for key, value in self.buffers.items():
                if len(value) == 0:
                    pass
                else:
                    # the samples in the first batch
                    samples = value[0]
                    if isinstance(samples, np.ndarray):
                        # value is a list of ndarray
                        value = np.concatenate(value, 0)
                    elif isinstance(samples, (list, tuple)):
                        # value is a list of lists of something (e.g. file path)
                        # flatten the inner list (stands for batch)
                        value = list(chain.from_iterable(value))
                    else:
                        raise TypeError(f'Invalid type of data stored in the buffer: {samples.__class__.__name__}.')
                    self.buffers.update({key: value})

        pred = self.buffers['pred']
        target = self.buffers['target']

        # filter the pred according to the pred_select_indices
        pred = pred[:, self.pred_select_inds]
        # filter the target according to the target_select_indices
        target = target[:, self.target_select_inds]

        img_files = self.buffers['img_file']
        df = pd.DataFrame(img_files, columns=['Image Index'])
        df[['p-' + x for x in self.target_select_names]] = pred
        df[['t-' + x for x in self.target_select_names]] = target
        for k, v in self.buffers.items():
            if k not in ['pred', 'target', 'img_file']:
                df[k] = v

        df.to_csv(self.file_path)
        self.logger.info(f'Predictions have been saved to {self.file_path}.')
        self._has_saved = True

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self._init_buffers)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._update_buffers)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._save_buffers)
