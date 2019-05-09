import os

import torch
from torch.utils.data.dataset import Dataset

from ..util.image import images_in_directory, imread, to_rgb


class ImageDirectory(Dataset):
    LABEL_FILENAME = 'labels.txt'

    def __init__(self,
                 root,
                 return_labels=True,
                 return_filenames=False,
                 transform=None):

        self.root = root
        self.transform = transform
        self.return_labels = return_labels
        self.return_filenames = return_filenames

        self._files = []
        self._labels = None
        self._get_paths()

    def __getitem__(self, index):
        filename = self._files[index]

        img = to_rgb(imread(os.path.join(self.root, filename)))

        if self.transform:
            img = self.transform(img)

        ret = [img]

        if self.return_labels:
            ret.append(torch.tensor([self._labels[index]]))

        if self.return_filenames:
            ret.append(filename)

        return ret[0] if len(ret) == 1 else ret

    def __len__(self):
        return len(self._files)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        if not os.path.isdir(root):
            fmt = "not a directory: '{}'"
            raise ValueError(fmt.format(root))

        self._root = root

    def _get_paths(self):
        # build list of image paths
        self._files = images_in_directory(self.root, exclude_root=True)

        if not self.return_labels:
            return

        # check whether label file exists
        labels_path = os.path.join(self._root, self.LABEL_FILENAME)

        if not os.path.exists(labels_path):
            self.return_labels = False
            return

        # if so, build list of labels
        label_dict = {}

        with open(labels_path, 'r') as f:
            for line in f:
                path, val = line.split()

                label_dict[path] = int(val)

        self._labels = [label_dict[path] for path in self._files]
