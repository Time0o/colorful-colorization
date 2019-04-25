from collections import OrderedDict
from time import time

import numpy as np
import torch
from torch import cuda

from .modules.colorization_network import ColorizationNetwork


_BATCH_DIMS_DEFAULT = (32, 3, 256, 256)

_L_SAFE = 50
_AB_SAFE = 0

_N_DRY_NETWORK_DEFAULT = 3
_N_STAT_NETWORK_DEFAULT = 5


class BenchmarkResults:
    DISPLAY_PRECISION = 3

    def __init__(self, descr, metrics):
        self.descr = descr

        self.metrics = metrics
        self.mean = {}
        self.std = {}

        for metric, values in metrics.items():
            self.mean[metric] = np.mean(values)
            self.std[metric] = np.std(values)

    def __repr__(self):
        res = []
        for metric in self.metrics:
            fmt = "{metric}: mean = {mean:.{p}f}, std = {std:.{p}f}"

            res.append(fmt.format(metric=metric,
                                  mean=self.mean[metric],
                                  std=self.std[metric],
                                  p=self.DISPLAY_PRECISION))

        return "{}:\n{}".format(self.descr, '\n'.join(res))


def _display_progress(info, i, i_max):
    beg = '' if i == 1 else '\r'
    end = '\n' if i == i_max else ''

    fmt = beg + "{}: {}/{}"
    msg = fmt.format(info, i, i_max)

    display_width = len(fmt.format(info, i_max, i_max))
    msg_ljusted = msg.ljust(display_width)

    print(msg_ljusted, end=end, flush=True)


def _network_execution_time(network, batch):
    # forward pass
    start = time()

    cuda.synchronize()
    out, _ = network(batch)
    cuda.synchronize()

    t_forward = time() - start

    # backward pass
    start = time()

    cuda.synchronize()
    out.backward(out)
    cuda.synchronize()

    t_backward = time() - start

    return t_forward, t_backward


def benchmark_network(batch_dims=_BATCH_DIMS_DEFAULT,
                      n_dry=_N_DRY_NETWORK_DEFAULT,
                      n_stat=_N_STAT_NETWORK_DEFAULT,
                      device='cuda',
                      verbose=False):

    benchmark_descr = "network forward/backward pass"

    if verbose:
        print(benchmark_descr + ":")

    # construct network
    network = ColorizationNetwork()
    network.train()

    # construct dummy batch
    n, _, h, w = batch_dims

    l = torch.full((n, 1, h, w), _L_SAFE)
    ab = torch.full((n, 2, h, w), _AB_SAFE)

    dummy_batch = torch.cat((l, ab), dim=1)

    # move model and batch to gpu
    network.to(device)
    dummy_batch = dummy_batch.to(device)

    # dry runs
    for i in range(1, n_dry + 1):
        if verbose:
            _display_progress("dry run", i, n_dry)

        _network_execution_time(network, dummy_batch)

    # run benchmark
    t_forward = []
    t_backward = []

    for i in range(1, n_stat + 1):
        if verbose:
            _display_progress("sample run", i, n_stat)

        t_f, t_b = _network_execution_time(network, dummy_batch)

        t_forward.append(t_f)
        t_backward.append(t_b)

    res = BenchmarkResults(
        descr=benchmark_descr,
        metrics=OrderedDict([
            ("forward pass runtime", t_forward),
            ("backward pass runtime", t_backward)
        ])
    )

    return res
