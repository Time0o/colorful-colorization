import os

import matplotlib.pyplot as plt
import numpy as np

from ..util.image import imread
from .io import get_filenames_by_label, \
                get_imagenet_plaintext_labels, \
                read_classification, \
                read_labels
from .plot import subplots


_SCATTER_S = 10


def _classification_performance(image_dir):
    labels = read_labels(image_dir)
    classified = read_classification(image_dir)

    filenames_by_label = get_filenames_by_label(labels)

    num_labels = max(filenames_by_label.keys()) + 1
    class_accuracies = np.full(num_labels, np.nan)

    for label, filenames in filenames_by_label.items():
        correct = 0
        for filename in filenames:
            if label in classified[filename]:
                correct += 1

        class_accuracies[label] = correct / len(filenames)

    return class_accuracies


def _confusion_matrix(image_dir):
    labels = read_labels(image_dir)
    classified = read_classification(image_dir)

    filenames_by_label = get_filenames_by_label(labels)

    num_labels = max(filenames_by_label.keys()) + 1
    c = np.zeros(((num_labels,) * 2))

    for ground_truth_label, filenames in filenames_by_label.items():
        for label in filenames_by_label:
            correct = 0
            for filename in filenames:
                if label in classified[filename]:
                    correct += 1

            c[ground_truth_label, label] = correct / len(filenames)

    return c


def gray_vs_recolorized_performance(no_color_dir,
                                    predict_color_dir,
                                    n_top=50,
                                    n_bottom=50,
                                    ax=None):

    a_no_color = _classification_performance(no_color_dir)
    a_predict_color = _classification_performance(predict_color_dir)

    a_rel_sorted = np.argsort(a_predict_color - a_no_color)

    # plot results
    if ax is None:
        _, ax = subplots(no_ticks=False)

    ax.scatter(a_no_color, a_predict_color, color='k', s=_SCATTER_S)

    if n_top > 0:
        ax.scatter(a_no_color[a_rel_sorted[-n_top:]],
                   a_predict_color[a_rel_sorted[-n_top:]],
                   facecolor='b',
                   s=_SCATTER_S,
                   label="Top {}".format(n_top))

    if n_bottom > 0:
        ax.scatter(a_no_color[a_rel_sorted[:n_bottom]],
                   a_predict_color[a_rel_sorted[:n_bottom]],
                   facecolor='r',
                   s=_SCATTER_S,
                   label="Bottom {}".format(n_bottom))

    ax.plot([0, 1], [0, 1], color='k')

    ax.set_title("VGG Classification Performance")
    ax.set_xlabel("Greyscale")
    ax.set_ylabel("Recolored")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if n_top > 0 or n_bottom > 0:
        ax.legend()


def top5_confusion_rates(ground_truth_dir,
                         predict_color_dir,
                         n_top=100,
                         ax=None):

    c_ground_truth = _confusion_matrix(ground_truth_dir)
    c_predict_color = _confusion_matrix(predict_color_dir)

    mask = ~np.eye(c_ground_truth.shape[0], dtype=bool)
    c_ground_truth = c_ground_truth[mask].flatten()
    c_predict_color = c_predict_color[mask].flatten()

    i = np.where((c_ground_truth == 0) & (c_predict_color == 0))
    c_ground_truth = np.delete(c_ground_truth, i)
    c_predict_color = np.delete(c_predict_color, i)

    a_rel_sorted = np.argsort(c_predict_color - c_ground_truth)

    # plot results
    if ax is None:
        _, ax = subplots(no_ticks=False)

    ax.scatter(c_ground_truth, c_predict_color, color='k', s=_SCATTER_S)

    ax.scatter(c_ground_truth[a_rel_sorted[-n_top:]],
               c_predict_color[a_rel_sorted[-n_top:]],
               facecolor='r',
               s=_SCATTER_S,
               label="Top {}".format(n_top))

    ax.plot([0, 1], [0, 1], color='k')

    ax.set_title("Confusion Rates")
    ax.set_xlabel("Greyscale")
    ax.set_ylabel("Recolored")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.legend()


def common_confusions(ground_truth_dir,
                      predict_color_dir,
                      which,
                      n_per_class=5,
                      axes=None):

    c_ground_truth = _confusion_matrix(ground_truth_dir)
    c_predict_color = _confusion_matrix(predict_color_dir)

    a = c_predict_color - c_ground_truth
    np.fill_diagonal(a, -np.inf)

    b = np.flip(np.argsort(a.ravel()))
    c_top, d_top = np.unravel_index(b, a.shape)

    labels = read_labels(ground_truth_dir)
    filenames_by_label = get_filenames_by_label(labels)

    classified_gt = read_classification(ground_truth_dir)
    classified_pc = read_classification(predict_color_dir)

    imagenet_plaintext_labels = get_imagenet_plaintext_labels()

    # determine "confused" images
    demo_images = []

    for filename in filenames_by_label[c_top[which]]:
        confused_gt = d_top[which] in classified_gt[filename]
        confused_pc = d_top[which] in classified_pc[filename]

        if not confused_gt and confused_pc:
            demo_images.append(filename)

            if len(demo_images) == n_per_class:
                break

    # plot results
    if axes is None:
        _, axes = subplots(2, n_per_class, use_gridspec=True, grid_spacing=0)

    for j, img in enumerate(demo_images):
        img_gt = imread(os.path.join(ground_truth_dir, img))
        img_pc = imread(os.path.join(predict_color_dir, img))

        axes[0, j].imshow(img_gt)
        axes[1, j].imshow(img_pc)

    # add plot text
    FONTSIZE = 15

    c_name = imagenet_plaintext_labels[c_top[which]]
    d_name = imagenet_plaintext_labels[d_top[which]]

    fmt = r"{} $\longrightarrow$ {}"
    plt.suptitle(fmt.format(c_name, d_name), fontsize=FONTSIZE)

    axes[0, 0].set_ylabel("Ground\nTruth", fontsize=FONTSIZE)
    axes[1, 0].set_ylabel("Recolored", fontsize=FONTSIZE)
