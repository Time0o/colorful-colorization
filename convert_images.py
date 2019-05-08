#!/usr/bin/env python3

import argparse
import random
from glob import glob
from shutil import copyfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from colorization.data.image_directory import ImageDirectory
from colorization.data.transforms import RGBOrGrayToL
from colorization.modules.colorization_network import ColorizationNetwork
from colorization.util.argparse import nice_help_formatter
from colorization.util.image import *


USAGE = \
"""convert_images.py [-h|--help]
                         --input-dir INPUT_DIR
                         --output-dir OUTPUT_DIR
                         [--color-source-dir COLOR_SOURCE_DIR]
                         [--annealed-mean-T ANNEALED_MEAN_T]
                         [--model-checkpoint MODEL_CHECKPOINT]
                         [--gpu]
                         [--verbose]
                         {resize,no_color,random_color,predict_color}"""


def _random_color(img, color_source_images):
    while True:
        color_source = imread(random.choice(color_source_images))

        if len(color_source.shape) == 3 and color_source.shape[2] == 3:
            break

    l = img[:, :, :1]
    ab = rgb_to_lab(color_source)[:, :, 1:]

    return lab_to_rgb(np.dstack((l, ab)))


def _predict_color(args):
    # create dataloader
    dataset = ImageDirectory(
        args.input_dir,
        return_labels=False,
        return_filenames=True,
        transform=transforms.Compose([
            RGBOrGrayToL(),
            transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    )

    dataloader = DataLoader(dataset, pin_memory=args.gpu)

    # create network
    device = 'cuda' if args.gpu else 'cpu'

    network = ColorizationNetwork(annealed_mean_T=args.annealed_mean_T,
                                  device=device)
    network.to(device)
    network.eval()

    # load checkpoint
    state = torch.load(args.model_checkpoint)
    network.load_state_dict(state['network'])

    # process images
    with torch.no_grad():
        for l_torch, filename in dataloader:
            if args.verbose:
                print("processing '{}'".format(filename[0]))

            l_torch = l_torch.to(device)

            l = torch_to_numpy(l_torch)
            ab = resize(torch_to_numpy(network(l_torch)), l.shape[:2])

            out_img = lab_to_rgb(np.dstack((l, ab)))
            out_path = os.path.join(args.output_dir, filename[0])

            imsave(out_path, out_img)


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=nice_help_formatter(),
                                     usage=USAGE)

    parser.add_argument('mode',
        choices=['resize', 'no_color', 'random_color', 'predict_color'])

    parser.add_argument('--input-dir',
                        required=True,
                        help="input image directory")

    parser.add_argument('--output-dir',
                        required=True,
                        help="output image directory")

    parser.add_argument('--resize-height',
                        type=int,
                        help=str("only meaningful in conjunction with "
                                 "'resize'"))

    parser.add_argument('--resize-width',
                        type=int,
                        help=str("only meaningful in conjunction with "
                                 "'resize'"))

    parser.add_argument('--color-source-dir',
                        help=str("only meaningful in conjunction with "
                                 "'random_color', random colors are extracted "
                                 "from randomly chosen images in this "
                                 "directory "))

    parser.add_argument('--annealed-mean-T',
                        type=float,
                        default=0.38,
                        help=str("only meaningful in conjunction with "
                                 "'predict_color', annealed mean decoding "
                                 "temperature parameter (default: %(default)s)"))

    parser.add_argument('--model-checkpoint',
                        help=str("only meaningful in conjunction with "
                                 "'predict_color', the color prediction "
                                 "model is loaded from this checkpoint"))

    parser.add_argument('--gpu',
                        action='store_true',
                        help=str("only meaningful in conjunction with "
                                 "'predict_color', run prediction on GPU"))

    parser.add_argument('--verbose',
                        action='store_true',
                        help="display progress")

    args = parser.parse_args()

    # validate arguments
    if args.mode == 'resize':
        if args.resize_height is None or args.resize_width is None:
            err = "dimensions must be specified if mode is 'resize'"
            raise ValueError(err)
    elif args.mode == 'random_color' and args.color_source_dir is None:
        err = "--color-source-dir must be specified if mode is 'random_color'"
        raise ValueError(err)
    elif args.mode == 'predict_color' and args.model_checkpoint is None:
        err = "--model-checkpoint must be specified if mode is 'predict_color'"
        raise ValueError(err)

    # create output directory
    if os.path.exists(args.output_dir):
        if os.listdir(args.output_dir):
            raise ValueError("'{}' already exists".format(args.output_dir))
    else:
        os.mkdir(args.output_dir)

    # convert images
    if args.mode == 'predict_color':
        _predict_color(args)
    else:
        if args.mode == 'resize':
            h, w = args.resize_height, args.resize_width
            transform = lambda img: resize(img, (h, w))
        elif args.mode == 'no_color':
            transform = lambda img: rgb_to_gray(img)
        elif args.mode == 'random_color':
            color_source_images = images_in_directory(args.color_source_dir)
            transform = lambda img: _random_color(img, color_source_images)

        for filename in images_in_directory(args.input_dir, exclude_root=True):
            if args.verbose:
                print("processing '{}'".format(filename))

            in_path = os.path.join(args.input_dir, filename)
            out_path = os.path.join(args.output_dir, filename)

            img = imread(in_path)
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = np.dstack((img,) * 3)

            imsave(out_path, transform(imread(in_path)))

    # copy labels
    label_file = os.path.join(args.input_dir, ImageDirectory.LABEL_FILENAME)

    if os.path.exists(label_file):
        out_label_file = os.path.join(
            args.output_dir, ImageDirectory.LABEL_FILENAME)

        copyfile(label_file, out_label_file)
