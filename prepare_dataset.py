#!/usr/bin/env python3

import os
import random
import sys
from argparse import ArgumentParser
from glob import glob
from shutil import move
from subprocess import CalledProcessError, check_call

from colorization.util.argparse import nice_help_formatter
from colorization.util.image import imread, imsave, resize


TRAIN_SUBDIR = 'train'
VAL_SUBDIR = 'val'
TEST_SUBDIR = 'test'

ALL_SUBDIRS = [
    TRAIN_SUBDIR,
    VAL_SUBDIR,
    TEST_SUBDIR
]

USAGE = \
"""usage: prepare_dataset.py [-h|--help]
                                 [--flatten]
                                 [--purge]
                                 [--clean]
                                 [--file-ext EXT]
                                 [--val-split VAL_SPLIT]
                                 [--test-split TEST_SPLIT]
                                 [--no-shuffle]
                                 [--create-lmdb]
                                 [--convert-imageset SCRIPT]
                                 DATA_DIR"""


def _is_processed(data_dir):
    exists = {d: False for d in ALL_SUBDIRS}

    for subdir in ALL_SUBDIRS:
        if os.path.exists(os.path.join(data_dir, subdir)):
            exists[subdir] = True

    if all(exists.values()):
        err = "directory seems to be processed, skipping split"
        print(err, file=sys.stderr)
        return True
    elif any(exists.values()):
        err = "directory is partially processed, manual cleanup required"
        print(err, file=sys.stderr)
        sys.exit(1)
    else:
        return True


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
    for f in os.listdir(data_dir):
        if not f.endswith('.' + file_ext):
            os.remove(os.path.join(data_dir, f))


def _split_dataset(data_dir,
                   file_ext,
                   val_split,
                   test_split,
                   clean=True,
                   shuffle=True,
                   resize_height=None,
                   resize_width=None):

    data_all = glob(os.path.join(data_dir, '*.' + file_ext))

    if shuffle:
        random.shuffle(data_all)

    num_val = int(val_split * len(data_all))
    num_test = int(test_split * len(data_all))
    num_train = len(data_all) - num_val - num_test

    data_train = data_all[:num_train]
    data_val = data_all[num_train:(num_train + num_val)]
    data_test = data_all[(num_train + num_val):]

    for files, subdir in (data_train, TRAIN_SUBDIR), \
                         (data_val, VAL_SUBDIR), \
                         (data_test, TEST_SUBDIR):

        subdir_path = os.path.join(data_dir, subdir)

        if not os.path.exists(subdir_path):
            os.mkdir(subdir_path)

        for f in files:
            if not clean and resize_height is None and resize_width is None:
                move(f, subdir_path)
            else:
                img = imread(f)

                if clean:
                    # remove non-RGB images
                    if len(img.shape) != 3 or img.shape[2] not in [3, 4]:
                        os.remove(f)

                    # remove alpha channels
                    if img.shape[2] == 4:
                        img = img[:, :, :3]

                img = resize(img, (resize_height, resize_width))
                imsave(os.path.join(subdir_path, os.path.basename(f)), img)

                os.remove(f)


def _create_lmdbs(data_dir,
                  convert_imageset,
                  resize_height=None,
                  resize_width=None):

    if args.convert_imageset is None:
        print("missing convert_imageset script location", file=sys.stderr)
        sys.exit(1)

    # create lmdb subdirectory
    path = os.path.join(data_dir, 'lmdb')

    if not os.path.exists(path):
        os.mkdir(path)

    # create one lmdb per subdirectory
    for subdir in ALL_SUBDIRS:
        # create dummy label text file
        dummy_labels = os.path.join(data_dir, subdir + '.txt')

        with open(dummy_labels, 'w') as f:
            f.write('\n'.join([
                '/{} 0'.format(os.path.basename(f))
                for f in os.listdir(os.path.join(data_dir, subdir))
            ]))

        # call conversion script
        lmdb_args = [args.convert_imageset]

        if resize_height is not None:
            lmdb_args += ['--resize_height', str(resize_height)]

        if resize_width is not None:
            lmdb_args += ['--resize_width', str(resize_width)]

        lmdb_args += [
            os.path.abspath(os.path.join(data_dir, subdir)),
            dummy_labels,
            os.path.join(data_dir, 'lmdb', subdir)
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

    parser.add_argument('--clean',
                        action='store_true',
                        help=str("convert images to RGB if possible and delete "
                                 "them otherwise"))

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

    # avoid processing data directories twice
    if not _is_processed(args.data_dir):
        # flatten and purge
        if args.flatten:
            _flatten_data_dir(args.data_dir, args.purge, args.file_ext)
        elif args.purge:
            _purge_toplevel(args.data_dir, args.file_ext)

        # split
        _split_dataset(args.data_dir,
                       args.file_ext,
                       args.val_split,
                       args.test_split,
                       shuffle=not args.no_shuffle,
                       resize_height=args.resize_height,
                       resize_width=args.resize_width)

    # create lmdbs
    if args.create_lmdb:
        if args.convert_imageset is None:
            print("missing convert_imageset script location", file=sys.stderr)
            sys.exit(1)

        _create_lmdbs(args.data_dir,
                      args.convert_imageset,
                      resize_height=args.resize_height,
                      resize_width=args.resize_width)
