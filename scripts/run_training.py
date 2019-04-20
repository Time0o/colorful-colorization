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
