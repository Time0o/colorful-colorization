import re
from itertools import chain

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from .util.image import rgb_to_lab


_FIGURE_WIDTH = 12
_FIGURE_SPACING = 0.05


def _subplots(r, c, figsize=None):
    display_width = (_FIGURE_WIDTH - (c - 1) * _FIGURE_SPACING) / c
    figure_height = r * display_width + (r - 1) * _FIGURE_SPACING

    fig = plt.figure(figsize=(_FIGURE_WIDTH, figure_height))

    gs = gridspec.GridSpec(r, c)
    gs.update(wspace=_FIGURE_SPACING, hspace=_FIGURE_SPACING)

    axes = np.empty((r, c), dtype=np.object)

    for r_ in range(r):
        for c_ in range(c):
            axes[r_, c_] = plt.subplot(gs[r_ * c + c_])

    for ax in axes.flatten():
        ax.axis('off')

    return fig, axes


def learning_curve_from_log(filename,
                            loss_regex,
                            smoothing_alpha=0.05,
                            ax=None):

    _, ax = _subplots()

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


def annealed_mean_demo(model, image_set, ts=None, verbose=False):
    t_orig = model.network.decode_q.T

    if ts is None:
        ts = [1, .77, .58, .38, .29, .14, 0]

    assert len(ts) % 2 == 1

    _, axes = _subplots(len(image_set), len(ts))

    for c, t in enumerate(ts):
        if verbose:
            print("running prediction for T = {}".format(t))

        model.network.decode_q.T = t

        for r, image in enumerate(image_set):
            image_pred = image.predict(model)

            axes[r, c].imshow(image_pred.get())

    model.network.decode_q.T = t_orig

    for i, (t, ax) in enumerate(zip(ts, axes[0, :])):
        suptitle = {
            0: "Mean",
            len(ts) // 2: "Annealed Mean",
            len(ts) - 1: "Mode"
        }.get(i, '')

        if t == 0:
            title_fmt = "{}\n$T\\rightarrow{}$"
        else:
            title_fmt = "{}\n$T={}$"

        ax.set_title(title_fmt.format(suptitle, t))


def good_vs_bad_demo(model_norebal,
                     model_rebal,
                     image_set_good,
                     image_set_bad,
                     verbose=False):

    assert len(image_set_good) % 2 == 1
    assert len(image_set_bad) % 2 == 1

    fig, axes = _subplots(len(image_set_good) + len(image_set_bad), 4)

    for r, image in enumerate(chain(image_set_good, image_set_bad)):
        if verbose:
            print("running prediction for image {}".format(r + 1))

        # input
        axes[r, 0].imshow(image.get('lab')[:, :, 0], cmap='gray')

        # prediction
        axes[r, 1].imshow(image.predict(model_norebal).get())

        # prediction (with class rebalancing)
        axes[r, 2].imshow(image.predict(model_rebal).get())

        # ground truth
        axes[r, 3].imshow(image.get())

    # plot divider
    trans = blended_transform_factory(
        fig.transFigure, axes[len(image_set_good) - 1, 0].transAxes)

    line = Line2D([0.1, 0.925],
                  [-0.025, -0.025],
                  color='k',
                  linestyle='--',
                  transform=trans)

    fig.lines.append(line)

    # add titles
    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Classification")
    axes[0, 2].set_title("Classification\n(w/ Rebalancing)")
    axes[0, 3].set_title("Ground Truth")
