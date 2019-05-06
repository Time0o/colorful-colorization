import os
import re
from functools import partial
from itertools import chain
from math import ceil
from operator import itemgetter

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from torchvision.models import vgg16

from .util.image import Image, ImageSet


_FIGURE_WIDTH = 12
_FIGURE_SPACING = 0.05


def _subplots(r, c, use_gridspec=False):
    display_width = (_FIGURE_WIDTH - (c - 1) * _FIGURE_SPACING) / c
    figure_height = r * display_width + (r - 1) * _FIGURE_SPACING

    if use_gridspec:
        fig = plt.figure(figsize=(_FIGURE_WIDTH, figure_height))

        gs = gridspec.GridSpec(r, c)
        gs.update(wspace=_FIGURE_SPACING, hspace=_FIGURE_SPACING)

        axes = np.empty((r, c), dtype=np.object)

        for r_ in range(r):
            for c_ in range(c):
                axes[r_, c_] = plt.subplot(gs[r_ * c + c_])
    else:
        fig, axes = plt.subplots(r, c, figsize=(_FIGURE_WIDTH, figure_height))
        axes = axes.reshape(r, c)

    for ax in axes.flatten():
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False)

    return fig, axes


def _bbox(fig, ax):
    r = fig.canvas.get_renderer()

    return ax.get_tightbbox(r).transformed(fig.transFigure.inverted())


def _subplot_divider(fig, axes, orientation, n):
    bbox = partial(_bbox, fig)

    line2d = partial(Line2D,
                     transform=fig.transFigure,
                     color='k',
                     linestyle='--')

    if orientation == 'horizontal':
        y = (bbox(axes[n, 0]).y1 + bbox(axes[n + 1, 0]).y0) / 2
        line = line2d([0, 1], [y, y])
    elif orientation == 'vertical':
        x = (bbox(axes[0, n]).x1 + bbox(axes[0, n + 1]).x0) / 2
        line = line2d([x, x], [0, 1])
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    fig.add_artist(line)


def _read_txt(root, txt, val_transform=id):
    result = []

    with open(os.path.join(root, txt), 'r') as f:
        for line in f:
            path, val = line.split()

            result.append((os.path.join(root, path), val_transform(val)))

    return result


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

    _, axes = _subplots(len(image_set), len(ts), use_gridspec=True)

    for c, t in enumerate(ts):
        if verbose:
            print("running prediction for T = {}".format(t))

        model.network.decode_q.T = t

        for r, image in enumerate(image_set):
            image_pred = image.predict_color(model)

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

    fig, axes = _subplots(
        len(image_set_good) + len(image_set_bad), 4, use_gridspec=False)

    for r, image in enumerate(chain(image_set_good, image_set_bad)):
        if verbose:
            print("running prediction for image {}".format(r + 1))

        # input
        axes[r, 0].imshow(image.get('lab')[:, :, 0], cmap='gray')

        # prediction
        axes[r, 1].imshow(image.predict_color(model_norebal).get())

        # prediction (with class rebalancing)
        axes[r, 2].imshow(image.predict_color(model_rebal).get())

        # ground truth
        axes[r, 3].imshow(image.get())

    # plot divider
    plt.tight_layout()

    _subplot_divider(fig, axes, 'horizontal', len(image_set_good) - 1)

    # add titles
    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Classification")
    axes[0, 2].set_title("Classification\n(w/ Rebalancing)")
    axes[0, 3].set_title("Ground Truth")


def amt_results_demo(model,
                     results_dir,
                     rows=4,
                     columns_best=3,
                     columns_worst=1,
                     verbose=False):

    # parse results
    txt = _read_txt(results_dir, 'results.txt', val_transform=float)
    results = {Image.load(path): val for path, val in txt}

    assert len(results) >= rows * (columns_worst + columns_best)

    images_sorted = [
        img for img, _ in sorted(results.items(), key=itemgetter(1))
    ]

    # plot results
    fig, axes = _subplots(rows, (columns_best + columns_worst) * 2)

    images_show = list(reversed(
        images_sorted[:(rows * columns_worst)] + \
        images_sorted[-(rows * columns_best):]
    ))

    for c in range(columns_best + columns_worst):
        for r in range(rows):
            n = c * rows + r

            if verbose:
                print("running prediction for image {}".format(n + 1))

            image = images_show[n]

            axes[r, 2 * c].imshow(image.get())
            axes[r, 2 * c + 1].imshow(image.predict_color(model).get())

            axes[r, 2 * c].set_ylabel("{}%".format(int(100 * results[image])))

    # plot divider
    plt.tight_layout()

    _subplot_divider(fig, axes, 'vertical', 2 * columns_best - 1)

    # add titles
    for i, ax in enumerate(axes[0, :]):
        if i % 2 == 0:
            ax.set_title("Ground Truth")
        else:
            ax.set_title("Ours")

    fmt = '{}' + ' ' * 140 + '{}'

    suptitle = fmt.format(r'Fooled more often $\longleftarrow$',
                          r'$\longrightarrow$ Fooled less often')

    fig.suptitle(suptitle, y=0)


def vgg_performance_demo(ctest10k_dir,
                         classification_model=None,
                         preprocessing_model=None,
                         batch_size=100,
                         verbose=False):

    # load model
    if classification_model is None:
         classification_model = vgg16(pretrained=True)

    # create image set
    txt = _read_txt(ctest10k_dir, 'ctest10k.txt', val_transform=int)
    paths, labels = zip(*txt)

    image_set = ImageSet.from_paths(paths)

    # run prediction
    labels_predicted = []

    for b in range(ceil(len(image_set) / batch_size)):
        batch_begin = b * batch_size
        batch_end = min((b + 1) * batch_size, len(image_set))

        if verbose:
            fmt = "running prediction for images {}...{}"
            print(fmt.format(batch_begin + 1, batch_end))

        batch = image_set[batch_begin:batch_end]

        labels_predicted += image_set.classify(classification_model,
                                               preprocessing_model)

    correct = np.count_nonzero(np.array(labels) == np.array(labels_predicted))

    return correct / len(image_set)
