import json
import os

from ..util.resources import get_resource_path


_LABELS_FILE = 'labels.txt'
_CLASSIFICATIONS_FILE = 'classifications.txt'
_IMAGENET_PLAINTEXT_LABELS = 'imagenet_plaintext_labels.json'


def read_lines(filename):
    with open(filename, 'r') as f:
        return [line[:-1] for line in f]


def read_labels(image_dir):
    ret = {}
    with open(os.path.join(image_dir, _LABELS_FILE), 'r') as f:
        for line in f:
            path, res = line.split()
            ret[path] = int(res)

    return ret


def read_classification(image_dir):
    ret = {}
    with open(os.path.join(image_dir, _CLASSIFICATIONS_FILE), 'r') as f:
        for line in f:
            tmp = line.split()
            ret[tmp[0]] = [int(t) for t in tmp[1:]]

    return ret


def get_filenames_by_label(labels):
    filenames_by_label = {}
    for filename, label in labels.items():
        filenames_by_label[label] = \
            filenames_by_label.get(label, []) + [filename]

    return filenames_by_label


def get_imagenet_plaintext_labels():
    with open(get_resource_path(_IMAGENET_PLAINTEXT_LABELS), 'r') as f:
        return {int(k): v for k, v in json.load(f).items()}
