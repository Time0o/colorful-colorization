import os
import warnings
from glob import glob
from mimetypes import types_map

import numpy as np
import torch
from skimage import color, io, transform


def image_extensions():
    extensions = []
    for ext, t in types_map.items():
        if t.split('/')[0] == 'image':
            extensions.append(ext[1:])

    return extensions


def images_in_directory(root):
    paths = []

    extensions = image_extensions()

    for path in glob(os.path.join(root, '*')):
        ext = path.rsplit('.')[-1].lower()

        if any([ext == ext_ for ext_ in extensions]):
            paths.append(path)

    return paths


def rgb_to_lab(img):
    assert img.dtype == np.uint8

    return color.rgb2lab(img).astype(np.float32)


def rgb_to_gray(img):
    assert img.dtype == np.uint8

    c = (255 * color.rgb2gray(img)).astype(np.uint8)

    return np.dstack((c, c, c))


def lab_to_rgb(img):
    assert img.dtype == np.float32

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        return (255 * np.clip(color.lab2rgb(img), 0, 1)).astype(np.uint8)


def numpy_to_torch(img):
    h, w, c = img.shape

    tensor = torch.from_numpy(np.moveaxis(img, -1, 0).reshape(1, c, h, w))

    return tensor.type(torch.float32)


def torch_to_numpy(batch):
    assert batch.shape[0] == 1

    return batch[0, :, :, :].cpu().numpy().transpose(1, 2, 0)


def resize(img, size):
    return transform.resize(img, size, mode='reflect', anti_aliasing=True)


def imread(path):
    return io.imread(path)


def imsave(path, img):
    if np.issubdtype(img.dtype, np.floating):
        img = 255 * img

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        io.imsave(path, img.astype(np.uint8))
