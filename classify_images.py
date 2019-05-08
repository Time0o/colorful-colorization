#!/usr/bin/env python3

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vgg16

from colorization.data.image_directory import ImageDirectory
from colorization.util.argparse import nice_help_formatter


USAGE = \
"""classify_images.py [-h|--help]
                          --image-dir IMAGE_DIR
                          [--top N]
                          [--gpu]
                          [--verbose]"""

RESULTFILE = "classified.txt"


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=nice_help_formatter(),
                                     usage=USAGE)

    parser.add_argument('--image-dir',
                        required=True,
                        help="directory containing images to be classified")

    parser.add_argument('--top',
                        metavar='N',
                        type=int,
                        default=5,
                        help="store top N classes (default: %(default)s)")

    parser.add_argument('--gpu',
                        action='store_true',
                        help="run classification on GPU")

    parser.add_argument('--verbose',
                        action='store_true',
                        help="display progress")

    args = parser.parse_args()

    # avoid reclassification
    if os.path.exists(os.path.join(args.image_dir, RESULTFILE)):
        print("refusing to reclassify", file=sys.stderr)
        sys.exit(1)

    # create dataloader
    dataset = ImageDirectory(
        args.image_dir,
        return_labels=True,
        return_filenames=True,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    )

    dataloader = DataLoader(dataset, pin_memory=args.gpu)

    # create network
    device = 'cuda' if args.gpu else 'cpu'

    network = vgg16(pretrained=True)
    network.to(device)
    network.eval() 

    # run classification
    results = []

    with torch.no_grad():
        for img, label, filename in dataloader:
            if args.verbose:
                print("processing '{}'".format(filename[0]))

            img = img.to(device)

            pred = network(img)

            _, top = pred.sort(descending=True)
            top = top.squeeze()[:args.top].tolist()

            results.append((filename[0], top))

    # write results file
    with open(os.path.join(args.image_dir, RESULTFILE), 'w') as f:
        for filename, top in results:
            f.write("{} {}\n".format(filename, ' '.join([str(t) for t in top])))
