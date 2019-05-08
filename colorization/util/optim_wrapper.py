from functools import partial

from torch import optim


class OptimWrapper:
    def __init__(self, optim_type, optim_params):
        self._optim = partial(self._get_optim(optim_type), **optim_params)

    def __call__(self, parameters):
        return self._optim(parameters)

    @staticmethod
    def _get_optim(optim_type):
        return getattr(optim, optim_type)
