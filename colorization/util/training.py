from functools import partial

from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from .resources import get_class


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, power, last_epoch=-1):
        self.max_epochs = None
        self.power = power

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.max_epochs is None:
            return [g['lr'] for g in self.optimizer.param_groups]

        return [
            g['lr'] * (1 - self.last_epoch / self.max_epochs)**self.power
            for g in self.optimizer.param_groups
        ]


class OptimWrapper:
    def __init__(self, optim_type, optim_params):
        self._optim = partial(get_class(optim_type), **optim_params)

    def __call__(self, parameters):
        return self._optim(parameters)


class LRSchedulerWrapper:
    def __init__(self, sched_type, sched_params):
        self._sched = partial(get_class(sched_type), **sched_params)

    def __call__(self, optim):
        return self._sched(optim)
