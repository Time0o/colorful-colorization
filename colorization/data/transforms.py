import random

import numpy as np
import torch

from ..util.image import images_in_directory, \
                         imread, \
                         lab_to_rgb, \
                         numpy_to_torch, \
                         resize, \
                         rgb_to_gray, \
                         rgb_to_lab, \
                         torch_to_numpy


class ToNumpy:
    def __call__(self, img):
        return np.array(img)


class RGBToGray:
    def __call__(self, img):
        return rgb_to_gray(img)


class RGBToLab:
    def __call__(self, img):
        return rgb_to_lab(img)


class RandomColor:
    def __init__(self, color_source_dir):
        self._color_source_images = images_in_directory(color_source_dir)

    def __call__(self, img):
        l = rgb_to_lab(img)[:, :, :1]

        random_image = imread(random.choice(self._color_source_images))
        random_ab = rgb_to_lab(random_image)[:, :, 1:]

        return lab_to_rgb(np.dstack((l, random_ab)))


class PredictColor:
    def __init__(self,
                 model,
                 input_size=224,
                 output_lab=False):

        self.model = model
        self.input_size = input_size
        self.output_lab = output_lab

    def __call__(self, img_rgb):
        img_lab = rgb_to_lab(img_rgb)

        img_rgb_resized = resize(img_rgb, (self.input_size,) * 2)
        img_lab_resized = rgb_to_lab(img_rgb_resized)

        l_batch = numpy_to_torch(img_lab_resized[:, :, :1])
        ab_pred = torch_to_numpy(self.model.predict(l_batch))

        ab_pred_resized = resize(ab_pred, img_rgb.shape[:2])
        img_lab = np.dstack((img_lab[:, :, :1], ab_pred_resized))

        if self.output_lab:
            return img_lab
        else:
            return lab_to_rgb(img_lab)
