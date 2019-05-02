import numpy as np
import torch
from skimage import color, io, transform


def rgb_to_lab(img):
    return color.rgb2lab(img)


def lab_to_rgb(img):
    img_float = np.clip(color.lab2rgb(img), 0, 1)
    img_uint8 = (255 * img_float).astype(np.uint8)

    return img_uint8


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

    io.imsave(path, img.astype(np.uint8))
