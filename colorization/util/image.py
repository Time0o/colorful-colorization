import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from skimage import color, io, transform

import torchvision.transforms as transforms


_INPUT_SIZE_DEFAULT = 224

_CLASSIFY_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(_INPUT_SIZE_DEFAULT),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def _classify(model, batch):
    batch = torch.stack([_CLASSIFY_TRANSFORM(img) for img in batch])

    model.eval()

    with torch.no_grad():
        return model(batch)


class NoPreprocessing:
    def __call__(self, img):
        return img.get()


class ToGrayscale:
    def __call__(self, img):
        return img.get(colorspace='gray')


class Colorize:
    def __init__(self, model):
        self.model = model

    def __call__(self, img):
        return img.predict_color(self.model).get()


class RandomColor:
    def __init__(self, color_source_dir):
        self.color_source_image_set = ImageSet.from_directory(color_source_dir)

    def __call__(self, img):
        color_source = random.choice(self.color_source_image_set)

        l = img.get(colorspace='lab')[:, :, :1]
        ab = color_source.get(colorspace='lab')[:, :, 1:]

        return np.dstack((l, ab))


class Image:
    def __init__(self, img_rgb, img_lab=None):
        self._img_rgb = img_rgb

        if img_lab is not None:
            self._img_lab = img_lab
        else:
            self._img_lab = rgb_to_lab(img_rgb)

    @classmethod
    def load(cls, path):
        return cls(imread(path))

    def save(self, path):
        imsave(path, self._img_rgb)

    def get(self, colorspace='rgb'):
        if colorspace == 'rgb':
            return self._img_rgb
        elif colorspace == 'gray':
            return rgb_to_gray(self._img_rgb)
        elif colorspace == 'lab':
            return self._img_lab
        else:
            raise ValueError("invalid colorspace")

    def show(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        ax.imshow(self._img_rgb)

        ax.axis('off')

    def predict_color(self, model, input_size=_INPUT_SIZE_DEFAULT):
        img_rgb_resized = resize(self._img_rgb, (input_size,) * 2)
        img_lab_resized = rgb_to_lab(img_rgb_resized)

        l_batch = numpy_to_torch(img_lab_resized[:, :, :1])
        ab_pred = torch_to_numpy(model.predict(l_batch))
        ab_pred_resized = resize(ab_pred, self._img_rgb.shape[:2])
        img_lab = np.dstack((self._img_lab[:, :, :1], ab_pred_resized))
        img_rgb = lab_to_rgb(img_lab)

        return self.__class__(img_rgb, img_lab)

    def classify(self,
                 classification_model,
                 preprocessing_model=None):

        if preprocessing_model is None:
            preprocessing_model = NoPreprocessing()

        img = numpy_to_torch(preprocessing_model(self))

        return _classify(classification_model, img).argmax().item()


class ImageSet:
    def __init__(self, images, lazy=False):
        self._images = images
        self._lazy = lazy

    @classmethod
    def from_directory(cls, root):
        if not os.path.isdir(root):
            raise ValueError("not a directory: '{}'".format(root))

        files = [os.path.join(root, f) for f in sorted(os.listdir(root))]

        return cls(files, lazy=True)

    @classmethod
    def from_paths(cls, paths):
        return cls(paths, lazy=True)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.__class__(self._images[i], lazy=self._lazy)

        if self._lazy:
            return Image.load(self._images[i])
        else:
            return self._images[i]

    def predict_color(self, model, input_size=_INPUT_SIZE_DEFAULT):
        return self.__class__(
            [img.predict_color(model, input_size=input_size) for img in self])

    def classify(self,
                 classification_model,
                 preprocessing_model=None):

        if preprocessing_model is None:
            preprocessing_model = NoPreprocessing()

        batch = self._batch(preprocessing_model)

        out = _classify(classification_model, batch)

        return list(out.argmax(dim=1).numpy())

    def _batch(self, preprocessing_model):
        return torch.cat(
            [numpy_to_torch(preprocessing_model(img)) for img in self])


def rgb_to_lab(img):
    return color.rgb2lab(img)


def rgb_to_gray(img):
    c = color.rgb2gray(img)

    return np.dstack((c, c, c))


def lab_to_rgb(img):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        io.imsave(path, img.astype(np.uint8))
