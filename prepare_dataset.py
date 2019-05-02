#!/usr/bin/env python3

import os
import random
import sys
from argparse import ArgumentParser
from glob import glob
from shutil import move
from subprocess import CalledProcessError, check_call

from colorization.util.argparse import nice_help_formatter


USAGE = \
"""usage: prepare_dataset.py [-h|--help]
                                 [--flatten]
                                 [--purge]
                                 [--file-ext EXT]
                                 [--val-split VAL_SPLIT]
                                 [--test-split TEST_SPLIT]
                                 [--no-shuffle]
                                 [--create-lmdb]
                                 [--convert-imageset SCRIPT]
                                 DATA_DIR"""


def _flatten_data_dir(data_dir, purge, file_ext):
    for root, subdirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.' + file_ext):
                move(os.path.join(root, f), data_dir)
            elif purge:
                os.remove(os.path.join(root, f))

        if not os.listdir(root):
            os.removedirs(root)


def _purge_toplevel(data_dir, file_ext):
    for f in os.listdir(datadir):
        if not f.endswith('.' + file_ext):
            os.remove(os.path.join(root, f))


def _split_dataset(data_dir, file_ext, shuffle, val_split, test_split):
    data_all = glob(os.path.join(data_dir, '*.' + file_ext))

    if shuffle:
        random.shuffle(data_all)

    num_val = int(val_split * len(data_all))
    num_test = int(test_split * len(data_all))
    num_train = len(data_all) - num_val - num_test

    data_train = data_all[:num_train]
    data_val = data_all[num_train:(num_train + num_val)]
    data_test = data_all[(num_train + num_val):]

    for files, subdir in (data_train, 'train'), \
                         (data_val, 'val'), \
                         (data_test, 'test'):

        path = os.path.join(data_dir, subdir)

        if not os.path.exists(path):
            os.mkdir(path)

        for f in files:
            move(f, path)


def _create_lmdbs(data_dir, convert_imageset):
    if args.convert_imageset is None:
        print("missing convert_imageset script location", file=sys.stderr)
        sys.exit(1)

    # create lmdb subdirectory
    path = os.path.join(data_dir, 'lmdb')

    if not os.path.exists(path):
        os.mkdir(path)

    # create one lmdb per subdirectory
    for subdir in 'train', 'val', 'test':
        # create dummy label text file
        dummy_labels = subdir + '.txt'

        with open(dummy_labels, 'w') as f:
            f.write('\n'.join([
                '/{} 0'.format(os.path.basename(f))
                for f in os.listdir(os.path.join(data_dir, subdir))
            ]))

        # call conversion script
        lmdb_subdir = os.path.join(data_dir, 'lmdb', subdir)

        lmdb_args = [
            args.convert_imageset,
            os.path.abspath(os.path.join(data_dir, subdir)),
            dummy_labels,
            lmdb_subdir
        ]

        try:
            check_call(lmdb_args)
        except CalledProcessError:
            print("failed to call lmdb creation script", file=sys.stderr)

        # remove dummy label text file again
        os.remove(dummy_labels)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=nice_help_formatter(),
                            usage=USAGE)

    parser.add_argument('data_dir',
                        metavar='DATA_DIR',
                        help="data directory")

    parser.add_argument('--flatten',
                        action='store_true',
                        help=str("recursively move images in data directory "
                                 "to toplevel before performing split"))

    parser.add_argument('--purge',
                        action='store_true',
                        help="remove non images before performing split")

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

    if args.resize_height is not None or args.resize_width is not None:
        raise ValueError('TODO')

    # flatten and purge
    if args.flatten:
        _flatten_data_dir(args.data_dir, args.purge, args.file_ext)
    elif args.purge:
        _purge_toplevel(args.data_dir, args.file_ext)

    # split
    _split_dataset(args.data_dir,
                   args.file_ext,
                   not args.no_shuffle,
                   args.val_split,
                   args.test_split)

    # create lmdbs
    if args.create_lmdb:
        if args.convert_imageset is None:
            print("missing convert_imageset script location", file=sys.stderr)
            sys.exit(1)

        _create_lmdbs(args.data_dir, args.convert_imageset)
