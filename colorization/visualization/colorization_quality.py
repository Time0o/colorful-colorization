import os
from functools import partial
from itertools import chain
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np

from ..cielab import DEFAULT_CIELAB
from ..util.image import images_in_directory, imread, rgb_to_lab
from ..util.progress import display_progress
from .io import read_classification, read_labels, read_lines
from .plot import subplot_divider, subplots


def _raw_accuracy(ab_pred, ab_label, thresh, reweigh_classes=False):
    if reweigh_classes:
        # bin ground truth pixels
        q = DEFAULT_CIELAB.bin_ab(ab_pred)

        # get pixel weights
        pixel_weights = 1 / DEFAULT_CIELAB.gamut.prior[q]
        pixel_weights /= pixel_weights.sum()

    # find pixels not exceeding threshold distance
    dist = np.linalg.norm(ab_label - ab_pred, axis=2)

    within_thresh = dist <= thresh
    num_within_thresh = np.count_nonzero(within_thresh)

    if num_within_thresh == 0:
        return 0

    if reweigh_classes:
        return (within_thresh * pixel_weights).sum()
    else:
        return num_within_thresh / np.prod(within_thresh.shape[:2])


def _auc(ab_pred,
         ab_label,
         thresh_min=0,
         thresh_max=151,
         thresh_step=10,
         reweigh_classes=False):

    thresh = np.arange(thresh_min, thresh_max, thresh_step)

    raw_acc = partial(_raw_accuracy,
                      ab_pred,
                      ab_label,
                      reweigh_classes=reweigh_classes)

    aucs = [raw_acc(t) for t in thresh]

    return sum(aucs) / len(thresh)


def good_vs_bad_demo(images_good_file,
                     images_bad_file,
                     image_dirs):

    image_paths_good = read_lines(images_good_file)
    image_paths_bad = read_lines(images_bad_file)

    fig, axes = subplots(len(image_paths_good) + len(image_paths_bad),
                         len(image_dirs),
                         use_gridspec=False)

    for r, image_path in enumerate(chain(image_paths_good, image_paths_bad)):
        for c, (image_dir, title) in enumerate(image_dirs):
            img = imread(os.path.join(image_dir, image_path))

            axes[r, c].imshow(img)

            if r == 0:
                axes[r, c].set_title(title)

    # plot divider
    plt.tight_layout()

    if image_paths_good and image_paths_bad:
        subplot_divider(fig, axes, 'horizontal', len(image_paths_good) - 1)


def amt_demo(participants,
             accuracies_file,
             ground_truth_dir,
             predict_color_dir,
             rows=4,
             columns_best=3,
             columns_worst=1,
             dirlabels=True):

    # parse results
    results = {}

    with open(accuracies_file, 'r') as f:
        for line in f:
            path, res = line.split()
            results[path] = 1 - float(res)

    assert len(results) >= rows * (columns_worst + columns_best)

    images_sorted = [
        path for path, _ in sorted(results.items(), key=itemgetter(1))
    ]

    # plot results
    fig, axes = subplots(
        rows, (columns_best + columns_worst) * 3 - 1, use_gridspec=True)

    images_show = list(reversed(
        images_sorted[:(rows * columns_worst)] + \
        images_sorted[-(rows * columns_best):]
    ))

    for c in range(columns_best + columns_worst):
        for r in range(rows):
            n = c * rows + r

            path_gt = os.path.join(ground_truth_dir, images_show[n])
            path_pc = os.path.join(predict_color_dir, images_show[n])

            ax_gt = axes[r, 3 * c]
            ax_pc = axes[r, 3 * c + 1]

            ax_gt.imshow(imread(path_gt))
            ax_pc.imshow(imread(path_pc))

            wrong = round(participants * results[images_show[n]])

            ax_gt.set_ylabel("{}/{}".format(wrong, participants))
            ax_gt.yaxis.labelpad = 0

    # plot divider
    subplot_divider(
        fig, axes, 'vertical', 3 * columns_best - 2, 3 * columns_best)

    # format plot
    for ax in axes[0, 0::3]:
        ax.set_title("Ground\nTruth")

    for ax in axes[0, 1::3]:
        ax.set_title("Ours")

    for ax in axes[:, 2::3].flatten():
        ax.axis('off')

    if dirlabels:
        fmt = '{}' + 5 * '\\ ' + '{}'

        suptitle = fmt.format(r'Fooled more often $\longleftarrow$',
                              r'$\longrightarrow$ Fooled less often')

        fig.suptitle(suptitle, y=-0.1)


def raw_accuracy_demo(ground_truth_dir,
                      predict_color_dir,
                      reweigh_classes=False,
                      verbose=False):

    image_paths = images_in_directory(predict_color_dir, exclude_root=True)

    auc_total = 0

    for i, image_path in enumerate(image_paths):
        if verbose:
            display_progress(i, len(image_paths))

        img_gt = imread(os.path.join(ground_truth_dir, image_path))
        img_pc = imread(os.path.join(predict_color_dir, image_path))

        auc_total += _auc(rgb_to_lab(img_gt)[:, :, 1:],
                          rgb_to_lab(img_pc)[:, :, 1:],
                          reweigh_classes=reweigh_classes)

    return auc_total / len(image_paths)


def vgg_accuracy_demo(image_dir):
    labels = read_labels(image_dir)
    classifications = read_classification(image_dir)

    correct = 0
    for filename, label in labels.items():
        if label in classifications[filename]:
            correct += 1

    return correct / len(labels)
