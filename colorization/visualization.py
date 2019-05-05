import re

import matplotlib.pyplot as plt
import numpy as np
from rdp import rdp
from scipy.stats import linregress


def learning_curve_from_log(filename,
                            loss_regex,
                            smoothing_alpha=0.05,
                            ax=None):

    if ax is None:
        _, ax = plt.subplots()

    # create loss curve
    losses = []
    with open(filename, 'r') as f:
        for line in f:
            m = re.search(loss_regex, line)

            if m is not None:
                loss = float(m.group(1))
                losses.append(loss)

    iterations = np.arange(1, len(losses) + 1)

    # create smoothed loss curve
    losses_smoothed = losses[:1]
    for loss in losses[1:]:
        loss_smoothed = smoothing_alpha * loss + \
                        (1 - smoothing_alpha) * losses_smoothed[-1]

        losses_smoothed.append(loss_smoothed)

    # convert to log
    losses = np.log(losses)
    losses_smoothed = np.log(losses_smoothed)

    # plot loss curves
    p = ax.plot(iterations, losses, alpha=.5)
    ax.plot(iterations, losses_smoothed, color=p[0].get_color())

    # format plot
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (Log)")

    ax.grid()
