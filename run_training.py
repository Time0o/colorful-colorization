#!/usr/bin/env python3

import argparse

import torch

import colorization.config as config
from colorization.util.argparse import nice_help_formatter


USAGE = \
"""run_training.py [-h|--help]
                       --config CONFIG
                       [--default-config CONFIG]
                       [--init-proto PROTOTXT]
                       [--init-model CAFFEMODEL]
                       [--random-seed SEED]"""


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=nice_help_formatter(),
                                     usage=USAGE)

    parser.add_argument('--config',
                        required=True,
                        help="training configuration JSON file")

    parser.add_argument('--default-config',
                        metavar='CONFIG',
                        help="training default configuration JSON file")

    parser.add_argument('--init-proto',
                        metavar='PROTOTXT',
                        help="weight init Caffe prototxt file")

    parser.add_argument('--init-model',
                        metavar='CAFFEMODEL',
                        help="weight init Caffe caffemodel file")

    parser.add_argument('--random-seed',
                        metavar='SEED',
                        type=int,
                        help="torch random seed")

    args = parser.parse_args()

    # set random seed
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    # load configuration file(s)
    cfg = config.get_config(args.config, args.default_config)

    # prepare dataset
    dataloader = config.dataloader_from_config(cfg)

    # create model
    model = config.model_from_config(cfg)

    # initialize model
    if (args.init_proto is None) != (args.init_model is None):
        err = "--init-proto and --init-model must be specified together"
        raise ValueError(err)

    model.network.init_from_caffe(args.init_proto, args.init_model)

    # run training
    model.train(dataloader, **cfg['training_args'])
