#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser
from glob import glob
from random import shuffle
from shutil import move
from subprocess import CalledProcessError, check_call

from colorization.util.argparse import nice_help_formatter


USAGE = \
"""usage: prepare_dataset.py [-h|--help]
                                 [--file-ext EXT]
                                 [--val-split VAL_SPLIT]
                                 [--test-split TEST_SPLIT]
                                 [--no-shuffle]
                                 [--create-lmdb]
                                 [--convert-imageset SCRIPT]
                                 DATA_DIR"""


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=nice_help_formatter(),
                            usage=USAGE)

    parser.add_argument('data_dir',
                        metavar='DATA_DIR',
                        help="data directory")

    parser.add_argument('--file-ext',
                        metavar='EXT',
                        default='jpg',
                        help="image file extension (default: %(default)s)")

    parser.add_argument('--val-split',
                        type=int,
                        default=0.1,
                        help=str("relative size of validation set (default: "
                                 "%(default).1f)"))

    parser.add_argument('--test-split',
                        type=int,
                        default=0.1,
                        help=("relative size of test set (default: "
                              "%(default).1f)"))

    parser.add_argument('--no-shuffle',
                        action='store_true',
                        help="don't shuffle data samples prior to split")

    parser.add_argument('--resize-height',
                        type=int,
                        help="image resize height")

    parser.add_argument('--resize-width',
                        type=int,
                        help="image resize width")

    parser.add_argument('--create-lmdb',
                        action='store_true',
                        help="additionally create lmdb files in given directory")

    parser.add_argument('--convert-imageset',
                        metavar='SCRIPT',
                        help=str("create_imageset script location (required if "
                                 "--create-lmdb is set"))

    args = parser.parse_args()

    data_all = glob(os.path.join(args.data_dir, '*.' + args.file_ext))

    if not args.no_shuffle:
        shuffle(data_all)

    num_val = int(args.val_split * len(data_all))
    num_test = int(args.test_split * len(data_all))
    num_train = len(data_all) - num_val - num_test

    data_train = data_all[:num_train]
    data_val = data_all[num_train:(num_train + num_val)]
    data_test = data_all[(num_train + num_val):]

    for subdir in 'train', 'val', 'test':
        path = os.path.join(args.data_dir, subdir)

        if not os.path.exists(path):
            os.mkdir(path)

    if args.create_lmdb:
        if args.convert_imageset is None:
            print("missing convert_imageset script location", file=sys.stderr)
            sys.exit(1)

        path = os.path.join(args.data_dir, 'lmdb')

        if not os.path.exists(path):
            os.mkdir(path)

    for data_files, subdir in (data_train, 'train'), \
                              (data_val, 'val'), \
                              (data_test, 'test'):

        subdir_ = os.path.join(args.data_dir, subdir)
        for f in data_files:
            move(f, subdir_)

        subdir_labels = subdir_ + '.txt'
        with open(subdir_labels, 'w') as f:
            f.write('\n'.join([
                '/{} 0'.format(os.path.basename(f))
                for f in data_files
            ]))

        if args.create_lmdb:
            lmdb_subdir = os.path.join(args.data_dir, 'lmdb', subdir)

            lmdb_args = [args.convert_imageset]

            if args.resize_height is not None:
                lmdb_args += ['--resize_height', str(args.resize_height)]

            if args.resize_width is not None:
                lmdb_args += ['--resize_width', str(args.resize_width)]

            lmdb_args += [os.path.abspath(subdir_), subdir_labels, lmdb_subdir]

            try:
                check_call(lmdb_args)
            except CalledProcessError: 
                print("failed to call lmdb creation script", file=sys.stderr)
