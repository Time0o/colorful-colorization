import re

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


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

            axes[r, c].axis('off')

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
