#!/usr/bin/env python3

import argparse

import setup_path

import colorization.config as config


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',
                        required=True,
                        help="training configuration JSON file")

    parser.add_argument('--default-config',
                        help="training default configuration JSON file")

    args = parser.parse_args()

    # load configuration file(s)
    cfg = config.get_config(args.config, args.default_config)

    # prepare dataset
    dataloader = config.dataloader_from_config(cfg)

    # create model
    model = config.model_from_config(cfg)

    # create logger
    if 'log_args' in cfg:
        logger = config.logger_from_config(cfg)
    else:
        logger = None

    # run training
    model.train(dataloader, **cfg['training_args'], logger=logger)
