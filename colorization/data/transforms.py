import numpy as np

from ..util.image import \
    lab_to_rgb, numpy_to_torch, resize, rgb_to_gray, rgb_to_lab, torch_to_numpy


class RGBToGray:
    def __call__(self, img):
        return rgb_to_gray(img)


class RGBToLab:
    def __call__(self, img):
        return rgb_to_lab(img)


class PredictColor:
    def __init__(self, model, input_size=224):
        self.model = model
        self.input_size = input_size

    def __call__(self, img_rgb):
        img_lab = rgb_to_lab(img_rgb)

        img_rgb_resized = resize(img_rgb, (self.input_size,) * 2)
        img_lab_resized = rgb_to_lab(img_rgb_resized)

        l_batch = numpy_to_torch(img_lab_resized[:, :, :1])
        ab_pred = torch_to_numpy(self.model.predict(l_batch))

        ab_pred_resized = resize(ab_pred, img_rgb.shape[:2])
        img_lab = np.dstack((img_lab[:, :, :1], ab_pred_resized))

        return lab_to_rgb(img_lab)
