import numpy as np
from skimage import color, transform


def rgb_to_lab(img):
    return color.rgb2lab(img)


def lab_to_rgb(img):
    img_float = np.clip(color.lab2rgb(img), 0, 1)
    img_uint8 = (255 * img_float).astype(np.uint8)

    return img_uint8


def torch_to_numpy(batch):
    assert batch.shape[0] == 1

    return batch[0, :, :, :].cpu().numpy().transpose(1, 2, 0)


def resize(img, size):
    return transform.resize(img, size, mode='reflect', anti_aliasing=True)
