#!/usr/bin/env python3

import argparse

import colorization.config as config
from colorization.util.argparse import nice_help_formatter


USAGE = \
"""run_training.py [-h|--help]
                       --config CONFIG
                       [--default-config CONFIG]"""


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

    args = parser.parse_args()

    # load configuration file(s)
    cfg = config.get_config(args.config, args.default_config)

    # prepare dataset
    dataloader = config.dataloader_from_config(cfg)

    # create logger
    logger = config.logger_from_config(cfg)

    # create model
    model = config.model_from_config(cfg)

    # run training
    if cfg['training_args'].pop('resume', False):
        checkpoint_epoch = model.restore_training(
            cfg['training_args']['checkpoint_dir'],
            cfg['training_args'].pop('checkpoint_epoch', None))

        model.train(dataloader,
                    epoch_init=(checkpoint_epoch + 1),
                    **cfg['training_args'])
    else:
        model.train(dataloader, **cfg['training_args'])
