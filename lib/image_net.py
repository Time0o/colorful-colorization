import os
import pickle
import re
from glob import glob

import numpy as np
from skimage import color, io, transform
from torch.utils.data.dataset import Dataset

from cielab import CIELAB


class TinyImageNet(Dataset):
    DATASET_TRAIN = 'train'
    DATASET_VAL = 'val'
    DATASET_TEST = 'test'

    IMAGE_SIZE_ACTUAL = 64

    COLOR_SPACE_RGB = 'rgb'
    COLOR_SPACE_LAB = 'lab'

    CLEAN_ASSUME = 'assume'
    CLEAN_SKIP = 'skip'
    CLEAN_PURGE = 'purge'

    def __init__(self,
                 root,
                 dataset=DATASET_TRAIN,
                 image_size=IMAGE_SIZE_ACTUAL,
                 dtype_float=np.float32,
                 dtype_int=np.int64,
                 labeled=True,
                 normalized=True,
                 cielab=CIELAB(),
                 color_space=COLOR_SPACE_LAB,
                 clean=CLEAN_ASSUME,
                 transform=None):

        self.set_root(root)
        self.set_dataset(dataset)
        self.set_image_size(image_size)
        self.set_dtype(dtype_float, dtype_int)
        self.set_labeled(labeled)
        self.set_normalized(normalized)
        self.set_cielab(cielab)
        self.set_color_space(color_space)

        self._build_indices()
        self._clean(clean)

    @staticmethod
    def restore(path):
        if not path.endswith('.pickle'):
            path += '.pickle'

        with open(path, 'rb') as f:
            return pickle.load(f)

    def __getitem__(self, index):
        if isinstance(index, slice):
            r = range(*index.indices(len(self._indices[self.dataset])))

            return [self._getitem(i) for i in r]
        else:
            return self._getitem(index)

    def __len__(self):
        return len(self._indices[self.dataset])

    def set_root(self, root):
        if not os.path.isdir(root):
            fmt = "not a directory: '{}'"
            raise ValueError(fmt.format(root))

        self.root = root

    def set_dataset(self, dataset):
        valid = [self.DATASET_TRAIN, self.DATASET_VAL, self.DATASET_TEST]

        if dataset not in valid:
            fmt = "dataset must be either of {}"
            raise ValueError(fmt.format(', '.join(valid)))

        self.dataset = dataset

    def set_image_size(self, image_size):
        assert image_size >= self.IMAGE_SIZE_ACTUAL

        self.image_size = image_size

    def set_dtype(self, dtype_float, dtype_int):
        self.dtype_float = dtype_float
        self.dtype_int = dtype_int

    def set_labeled(self, labeled):
        self.labeled = labeled

    def set_normalized(self, normalized):
        self.normalized = normalized

    def set_cielab(self, cielab):
        self.cielab = cielab

    def set_color_space(self, color_space):
        if color_space not in [self.COLOR_SPACE_RGB, self.COLOR_SPACE_LAB]:
            raise ValueError("invalid color space")

        self.color_space = color_space

    def cache(self, path):
        if not path.endswith('.pickle'):
            path += '.pickle'

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def _build_indices(self):
        self._indices = {}

        for dataset in self.DATASET_TRAIN, self.DATASET_VAL, self.DATASET_TEST:
            self._build_index(dataset)

    def _build_index(self, dataset):
        self._indices[dataset] = []

        dataset_path = os.path.join(self.root, dataset)

        if dataset == self.DATASET_TRAIN:
            for images in self._listdir(dataset_path):
                images_root = os.path.join(images, 'images')

                for image_path in self._listdir(images_root, sort_num=True):
                    self._indices[dataset].append(image_path)
        else:
            images_root = os.path.join(dataset_path, 'images')
            self._indices[dataset] = self._listdir(images_root, sort_num=True)

    def _clean(self, clean):
        if clean == self.CLEAN_SKIP:
            self._filter_non_rgb()
        elif clean == self.CLEAN_PURGE:
            self._filter_non_rgb(purge=True)
        elif clean != self.CLEAN_ASSUME:
            raise ValueError("invalid cleaning procedure")

    def _filter_non_rgb(self, purge=False):
        for dataset, index in self._indices.items():
            index_rgb_only = []

            for i, path in enumerate(index):
                if self._is_rgb(io.imread(path)):
                    index_rgb_only.append(path)
                elif purge:
                    os.remove(path)

            self._indices[dataset] = index_rgb_only

    def _getitem(self, index):
        image_path = self._indices[self.dataset][index]
        image_rgb = io.imread(image_path)

        assert self._is_rgb(image_rgb) and self._has_right_size(image_rgb)

        # scale image to desired size
        if self.image_size != self.IMAGE_SIZE_ACTUAL:
            image_rgb = transform.pyramid_expand(
                image_rgb,
                upscale=(self.image_size / self.IMAGE_SIZE_ACTUAL),
                multichannel=True)

        if self.color_space == self.COLOR_SPACE_RGB:
            if self.labeled:
                raise ValueError("can not produce labeled data from RGB images")

            return self._process_image(image_rgb)

        elif self.color_space == self.COLOR_SPACE_LAB:
            image_lab = self.cielab.rgb_to_lab(image_rgb)

            if self.labeled:
                l, ab = self.cielab.dissemble(image_lab, expand_q=True)

                if self.normalized:
                    l = (l - 50) / 50

                return self._process_image(l), self._process_image(ab)
            else:
                return self._process_image(image_lab)

    def _process_image(self, image):
        assert len(image.shape) == 2 or len(image.shape) == 3

        if np.issubdtype(image.dtype, np.integer):
            dtype = self.dtype_int
        else:
            dtype = self.dtype_float

        image = image.astype(dtype)

        if len(image.shape) == 2:
            return image[np.newaxis]
        else:
            return np.moveaxis(image, -1, 0)

    @staticmethod
    def _listdir(path, sort_num=False):
        files = glob(os.path.join(path, '*'))

        if sort_num:
            def parse_num(f):
                base = f.rsplit('.', 1)[0]

                i = re.search(r'\d+$', base).start()
                base, num = base[:i], base[i:]

                return base, int(num)

            files.sort(key=parse_num)
        else:
            files.sort()

        return files

    @staticmethod
    def _is_rgb(image):
        return len(image.shape) == 3 and image.shape[2] == 3

    @classmethod
    def _has_right_size(cls, image):
        return image.shape[0] == image.shape[1] == cls.IMAGE_SIZE_ACTUAL
