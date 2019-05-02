import os
import re
from glob import glob

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToPILImage

from ..util.image import imread, rgb_to_lab


class StandardDataset(Dataset):
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
                 crop_size=None,
                 transform=None):

        self.root = root
        self.dataset = dataset
        self.limit = limit
        self.transform = transform

        self._build_indices()
        self._clean(clean)

    def __getitem__(self, index):
        path = self._indices[self.dataset][index]

        return self._load_and_process_image(path)

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
            self._indices[dataset] = []

            dataset_path = os.path.join(self.root, dataset)

            for image_path in self._listdir(dataset_path):
                self._indices[dataset].append(image_path)

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
                if self._is_rgb(imread(path)):
                    index_rgb_only.append(path)
                elif purge:
                    os.remove(path)

            self._indices[dataset] = index_rgb_only

    def _load_and_process_image(self, path):
        img_rgb = imread(path)

        if self.transform:
            img_pil = ToPILImage()(img_rgb)
            img_pil = self.transform(img_pil)
            img_rgb = np.array(img_pil)

        img_lab = rgb_to_lab(img_rgb)

        return np.moveaxis(img_lab.astype(self.DTYPE), -1, 0)

    @staticmethod
    def _listdir(path):
        files = glob(os.path.join(path, '*'))

        def parse_num(f):
            base = f.rsplit('.', 1)[0]

            i = re.search(r'\d+$', base).start()
            base, num = base[:i], base[i:]

            return base, int(num)

        files.sort(key=parse_num)

        return files
