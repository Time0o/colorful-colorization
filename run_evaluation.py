#!/usr/bin/env python3

import argparse
import os
from glob import glob

from skimage import color, io
from torch.utils.data import DataLoader

import colorization.config as config
from colorization.util.argparse import nice_help_formatter


USAGE = \
"""run_evaluation.py [-h|--help]
                         --config CONFIG
                         --output-dir DIR
                         [--checkpoint CHECKPOINT]
                         [--checkpoint-dir DIR]
                         [--batch-size BATCH_SIZE]"""


if __name__ == '__main__':
    # parse command line arguments
    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=40)

    parser = argparse.ArgumentParser(formatter_class=nice_help_formatter(),
                                     usage=USAGE)

    parser.add_argument('--config',
                        required=True,
                        help="evaluation configuration JSON file")

    parser.add_argument('--output-dir',
                        metavar='DIR',
                        required=True,
                        help=str("directory into which predicted color images "
                                 "will be written"))

    parser.add_argument('--checkpoint',
                        metavar='CHECKPOINT',
                        help=str("specific model checkpoint file from which to "
                                 "load model for prediction"))

    parser.add_argument('--checkpoint-dir',
                        metavar='DIR',
                        help=str("model checkpoint directory, if this is given "
                                 "and --model-checkpoint is not, the model "
                                 "corresponding to the most recent checkpoint "
                                 "is used to perform prediction"))

    args = parser.parse_args()

    # load configuration file(s)
    cfg = config.get_config(args.config)

    # create model
    model = config.model_from_config(cfg, trainable=False)

    if args.checkpoint is None:
        if args.checkpoint_dir is None:
            err = "either checkpoint path or directory must be given"
            raise ValueError(err)

        checkpoint_path, _ = model.find_latest_checkpoint(
            args.checkpoint_dir)
    else:
        checkpoint_path = args.checkpoint_path

    model.load(checkpoint_path)

    # run prediction
    dataloader = config.dataloader_from_config(cfg)

    for img_batch, paths in dataloader:
        img_pred_batch = model.predict(img_batch[:, :1, :, :])

        for img_pred, path in zip(img_pred_batch, paths):
            img_pred_numpy = img_pred.permute(1, 2, 0).numpy()
            img_pred_rgb = color.lab2rgb(img_pred_numpy)

            out_path = os.path.join(args.output_dir, os.path.basename(path))

            io.imsave(out_path, img_pred_rgb)
