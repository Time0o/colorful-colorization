import os
import re
from glob import glob

import numpy as np
from skimage import io
from torch.utils.data.dataset import Dataset

from ..util.image import rgb_to_lab


class TinyImageNet(Dataset):
    DATASET_TRAIN = 'train'
    DATASET_VAL = 'val'
    DATASET_TEST = 'test'

    DTYPE = np.float32

    CLEAN_ASSUME = 'assume'
    CLEAN_SKIP = 'skip'
    CLEAN_PURGE = 'purge'

    def __init__(self,
                 root,
                 dataset=DATASET_TRAIN,
                 limit=None,
                 clean=CLEAN_ASSUME,
                 transform=None):

        self.root = root
        self.dataset = dataset
        self.limit = limit
        self.transform = transform

        self._build_indices()
        self._clean(clean)

    def __getitem__(self, index):
        image_path = self._indices[self.dataset][index]
        img = io.imread(image_path)

        return self._process_image(img), image_path

    def __len__(self):
        l = len(self._indices[self.dataset])

        if self.limit is None:
            return l
        else:
            return min(self.limit, l)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        if not os.path.isdir(root):
            fmt = "not a directory: '{}'"
            raise ValueError(fmt.format(root))

        self._root = root

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        valid = [self.DATASET_TRAIN, self.DATASET_VAL, self.DATASET_TEST]

        if dataset not in valid:
            fmt = "dataset must be either of {}"
            raise ValueError(fmt.format(', '.join(valid)))

        self._dataset = dataset

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

    def _process_image(self, img):
        img = rgb_to_lab(img).astype(self.DTYPE)

        if self.transform:
            img = self.transform(img)

        return np.moveaxis(img, -1, 0)

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
