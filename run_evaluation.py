#!/usr/bin/env python3

import argparse
import os
import warnings
from glob import glob

import numpy as np
from skimage import color, io
from torch.utils.data import DataLoader

import colorization.config as config
from colorization.data.image_file_or_directory import ImageFileOrDirectory
from colorization.util.argparse import nice_help_formatter
from colorization.util.image import lab_to_rgb, resize, torch_to_numpy


USAGE = \
"""run_evaluation.py [-h|--help]
                         --input-image IMAGE
                         --output-image IMAG
                         [--pretrain-proto PROTOTXT]
                         [--pretrain-model CAFFEMODEL]
                         [--checkpoint CHECKPOINT]
                         [--checkpoint-dir DIR]"""


if __name__ == '__main__':
    # parse command line arguments
    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=40)

    parser = argparse.ArgumentParser(formatter_class=nice_help_formatter(),
                                     usage=USAGE)

    parser.add_argument('--input-image',
                        required=True,
                        metavar='IMAGE',
                        help="path to single input image")

    parser.add_argument('--output-image',
                        required=True,
                        metavar='IMAGE',
                        help=str("location to which predicted color image is "
                                 "to be written, only meaningful in "
                                 "combination with --input-image"))

    parser.add_argument('--pretrain-proto',
                        metavar='PROTOTXT',
                        help="Caffe prototxt file")

    parser.add_argument('--pretrain-model',
                        metavar='CAFFEMODEL',
                        help="Caffe caffemodel file")

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

    # create dataset
    if args.input_image is not None:
        if args.output_image is None:
            err = "--input-image and --output-image must be specified together"
            raise ValueError(err)

        dataset = ImageFileOrDirectory(args.input_image)

    dataloader = DataLoader(dataset)

    # create model
    model = config.model_from_config({}, trainable=False)

    # load pretrained weights
    if (args.pretrain_proto is None) != (args.pretrain_model is None):
        err = "--pretrain-proto and --pretrain-model must be specified together"
        raise ValueError(err)

    pretrained = args.pretrain_proto is not None

    if pretrained:
        model.network.init_from_caffe(args.pretrain_proto, args.pretrain_model)

    # load from checkpoint
    if (args.checkpoint is not None) and (args.checkoint_dir is not None):
        err = "--checkpoint and --checkpoint-dir may not be specified together"
        raise ValueError(err)

    checkpointed = (args.checkpoint is not None) or \
                   (args.checkpoint_dir is not None)

    if pretrained:
        if checkpointed:
            raise ValueError("can't load from pretrained model AND checkpoint")
    else:
        if not checkpointed:
            raise ValueError("no network specified")

    if checkpointed:
        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint
        elif args.checkpoint_dir is not None:
            checkpoint_path, _ = model.find_latest_checkpoint(args.checkpoint_dir)

        model.load(checkpoint_path)

    # run prediction
    for img in dataloader:
        # extract lightness channel
        l_in = img[:, :1, :, :]

        # run prediction
        l_out = torch_to_numpy(l_in)
        ab_out = torch_to_numpy(model.predict(img[:, :1, :, :]))

        # scale output
        ab_out = resize(ab_out, l_out.shape[:2])

        # assemble color image and transform to RGB colorspace
        img_out = np.dstack((l_out, ab_out))
        img_out_rgb = lab_to_rgb(img_out)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            io.imsave(args.output_image, img_out_rgb)
