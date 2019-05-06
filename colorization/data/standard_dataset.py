import os
from glob import glob

import numpy as np
from torch.utils.data.dataset import Dataset

from ..util.image import imread


class StandardDataset(Dataset):
    DATASET_TRAIN = 'train'
    DATASET_VAL = 'val'
    DATASET_TEST = 'test'

    DTYPE = np.float32

    def __init__(self,
                 root,
                 dataset=DATASET_TRAIN,
                 transform=None):

        self.root = root
        self.dataset = dataset
        self.transform = transform

        self._build_indices()

    def __getitem__(self, index):
        img = imread(self._indices[self.dataset][index])

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self._indices[self.dataset])

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

            for image_path in glob(os.path.join(dataset_path, '*')):
                self._indices[dataset].append(image_path)
