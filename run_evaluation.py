#!/usr/bin/env python3

import argparse
import os
from glob import glob

import numpy as np

import colorization.config as config
from colorization.util.argparse import nice_help_formatter
from colorization.util.image import \
    imread, imsave, lab_to_rgb, numpy_to_torch, resize, rgb_to_lab, torch_to_numpy


USAGE = \
"""run_evaluation.py [-h|--help]
                         --config CONFIG
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

    parser.add_argument('--config',
                        required=True,
                        help="training configuration JSON file")

    parser.add_argument('--input-image',
                        required=True,
                        metavar='IMAGE',
                        help="path to single input image")

    parser.add_argument('--output-image',
                        required=True,
                        metavar='IMAGE',
                        help=str("location to which predicted color image is "
                                 "to be written"))

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

    # load configuration file(s)
    cfg = config.get_config(args.config)
    cfg = config.parse_config(cfg)

    model = cfg['model']

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

    # load input image
    img_rgb = imread(args.input_image)
    img_rgb_resized = resize(img_rgb, (model.network.INPUT_SIZE,) * 2)

    h_orig, w_orig, _ = img_rgb.shape

    # convert colorspace
    img_lab = rgb_to_lab(img_rgb)
    img_lab_resized = rgb_to_lab(img_rgb_resized)

    # run prediction
    l_batch = numpy_to_torch(img_lab_resized[:, :, :1])
    ab_pred = torch_to_numpy(model.predict(l_batch))

    # assemble and save
    img_lab_pred = np.dstack(
        (img_lab[:, :, :1], resize(ab_pred, (h_orig, w_orig))))

    imsave(args.output_image, lab_to_rgb(img_lab_pred))
