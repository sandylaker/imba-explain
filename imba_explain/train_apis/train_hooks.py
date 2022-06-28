from datetime import timedelta
from logging import Logger
from typing import Optional

from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.utils import setup_logger
from torch.optim import Optimizer


class TrainStatsTextLogger:

    def __init__(self, interval: int = 1, logger: Optional[Logger] = None) -> None:
        self.interval = interval
        self.logger = logger if logger is not None else setup_logger('imba-explain')
        self.timer = Timer(average=True)

    def _log_stats(self, engine: Engine, optimizer: Optimizer) -> None:
        loss = engine.state.output
        iter_time = self.timer.value()
        max_iters = engine.state.max_epochs * engine.state.epoch_length
        remain_iters = max(0, max_iters - engine.state.iteration)
        eta_str = str(timedelta(seconds=int(remain_iters * iter_time)))

        log_str = f'Epoch [{engine.state.epoch}/{engine.state.max_epochs}] '
        iter_in_epoch = engine.state.iteration % engine.state.epoch_length
        # replace [0/epoch_length] with [epoch_length/epoch_length]
        iter_in_epoch = iter_in_epoch if iter_in_epoch != 0 else engine.state.epoch_length
        log_str += f'Iteration [{iter_in_epoch}/{engine.state.epoch_length}]: '
        log_str += f'batch time: {iter_time:.4f}, eta: {eta_str}, '
        log_str += f"lr: {optimizer.param_groups[0]['lr']:.2e}, "
        log_str += f'loss: {loss:.4f} '
        self.logger.info(log_str)

    def attach(self, engine: Engine, optimizer: Optimizer) -> None:
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self._log_stats, optimizer)
        self.timer.attach(
            engine,
            start=Events.EPOCH_STARTED(once=1),
            resume=Events.ITERATION_STARTED,
            pause=Events.ITERATION_COMPLETED,
            step=Events.ITERATION_COMPLETED)
