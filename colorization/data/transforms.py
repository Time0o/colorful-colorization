import numpy as np

from ..util.image import is_rgb, rgb_to_lab


class ToNumpy:
    def __call__(self, img):
        return np.array(img)


class RGBToLab:
    def __call__(self, img):
        return rgb_to_lab(img)


class RGBOrGrayToL:
    def __call__(self, img):
        if is_rgb(img):
            return rgb_to_lab(img)[:, :, :1]
        else:
            return img.reshape(*img.shape[:2], 1)
