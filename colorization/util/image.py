import os
import warnings
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


def images_in_directory(root, exclude_root=False):
    if not os.path.exists(root):
        raise ValueError("directory '{}' does not exist".format(root))

    paths = []

    extensions = image_extensions()

    for path in sorted(os.listdir(root)):
        ext = path.rsplit('.')[-1].lower()

        if any([ext == ext_ for ext_ in extensions]):
            if exclude_root:
                paths.append(path)
            else:
                paths.append(os.path.join(root, path))

    if not paths:
        raise ValueError("directory '{}' does not contain images".format(root))

    return paths


def is_rgb(img):
    return len(img.shape) == 3 and img.shape[2] == 3


def to_rgb(img):
    if is_rgb(img):
        return img
    else:
        return np.dstack((img,) * 3)


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
    res = transform.resize(img, size, mode='reflect', anti_aliasing=True)

    if img.dtype == np.uint8:
        res *= 255

    return res.astype(img.dtype)


def imread(path):
    return io.imread(path)


def imsave(path, img):
    if np.issubdtype(img.dtype, np.floating):
        img = 255 * img

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        io.imsave(path, img.astype(np.uint8))


def predict_color(model, img):
    if is_rgb(img):
        l = rgb_to_lab(img)[:, :, :1]
    else:
        l = img.reshape(*img.shape[:2], 1)

    l_torch = numpy_to_torch(l)
    ab = resize(torch_to_numpy(model.predict(l_torch)), l.shape[:2])

    return lab_to_rgb(np.dstack((l, ab)))
