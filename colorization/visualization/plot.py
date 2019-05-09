from functools import partial

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def subplots(r=1,
             c=1,
             use_gridspec=False,
             figure_width=12,
             grid_spacing=0.05,
             no_ticks=True):

    display_width = (figure_width - (c - 1) * grid_spacing) / c
    figure_height = r * display_width + (r - 1) * grid_spacing

    if use_gridspec:
        fig = plt.figure(figsize=(figure_width, figure_height))

        gs = gridspec.GridSpec(r, c)
        gs.update(wspace=grid_spacing, hspace=grid_spacing)

        axes = np.empty((r, c), dtype=np.object)

        for r_ in range(r):
            for c_ in range(c):
                axes[r_, c_] = plt.subplot(gs[r_ * c + c_])
    else:
        fig, axes = plt.subplots(r, c, figsize=(figure_width, figure_height))

        if not (r == 1 and c == 1):
            axes = axes.reshape(r, c)

    if no_ticks:
        for ax in ([axes] if r == c == 1 else axes.flatten()):
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


def bbox(fig, ax):
    r = fig.canvas.get_renderer()

    return ax.get_tightbbox(r).transformed(fig.transFigure.inverted())


def subplot_divider(fig, axes, orientation, n):
    bbox_ = partial(bbox, fig)

    line2d = partial(Line2D,
                     transform=fig.transFigure,
                     color='k',
                     linestyle='--')

    if orientation == 'horizontal':
        y = (bbox_(axes[n, 0]).y1 + bbox_(axes[n + 1, 0]).y0) / 2
        line = line2d([0, 1], [y, y])
    elif orientation == 'vertical':
        x = (bbox_(axes[0, n]).x1 + bbox_(axes[0, n + 1]).x0) / 2
        line = line2d([x, x], [0, 1])
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    fig.add_artist(line)
