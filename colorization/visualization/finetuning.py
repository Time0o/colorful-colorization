import os

import torch

from ..modules.cross_entropy_loss_2d import CrossEntropyLoss2d
from ..util.image import images_in_directory, imread, numpy_to_torch, rgb_to_lab
from .plot import subplots
from .progress import display_progress


def finetuning_loss_comparison(model,
                               model_finetuned,
                               image_dir,
                               batch_size=1,
                               subsample=None,
                               ax=None,
                               verbose=False):

    assert model.device == model_finetuned.device

    image_paths = images_in_directory(image_dir)
    if subsample is not None:
        image_paths = image_paths[:subsample]

    cross_entropy = CrossEntropyLoss2d()
    loss = lambda net, lab: cross_entropy(*net.forward_encode(lab)).item()

    losses = []
    losses_finetuned = []

    model.network.eval()
    model_finetuned.network.eval()

    with torch.no_grad():
        num_batches = len(image_paths) // batch_size

        for i in range(0, num_batches):
            if verbose:
                display_progress(i, num_batches, msg='processing batch')

            tensors = []
            for j in range(i * batch_size, (i + 1) * batch_size):
                img_lab = rgb_to_lab(imread(image_paths[j]))
                tensors.append(numpy_to_torch(img_lab))

            batch = torch.cat(tensors).to(model.device)

            losses.append(loss(model.network, batch))
            losses_finetuned.append(loss(model_finetuned.network, batch))

    if ax is None:
        _, ax = subplots(no_ticks=False)

    ax.hist(losses, alpha=0.5, label="Pretrained")
    ax.hist(losses_finetuned, alpha=0.5, label="Finetuned")

    ax.legend()
    ax.grid()

    ax.set_xlabel("Cross Entropy Loss")


def finetuning_demo(ground_truth_dir,
                    predict_color_dir,
                    predict_color_finetuned_dir,
                    n=10,
                    label_images=False):

    image_paths = images_in_directory(ground_truth_dir, exclude_root=True)

    _, axes = subplots(n, 3, use_gridspec=True)

    for i, image_path in enumerate(image_paths[:n]):
        img_gt = imread(
            os.path.join(ground_truth_dir, image_path))

        img_pred = imread(
            os.path.join(predict_color_dir, image_path))

        img_pred_finetuned = imread(
            os.path.join(predict_color_finetuned_dir, image_path))

        axes[i, 0].imshow(img_pred)
        axes[i, 1].imshow(img_pred_finetuned)
        axes[i, 2].imshow(img_gt)

        if label_images:
            axes[i, 0].set_ylabel(image_path)

    axes[0, 0].set_title("Pretrained")
    axes[0, 1].set_title("Finetuned")
    axes[0, 2].set_title("Groud Truth")
