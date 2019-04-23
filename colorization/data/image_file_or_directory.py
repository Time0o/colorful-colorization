import os

import numpy as np
from skimage import io
from torch.utils.data.dataset import Dataset

from ..util.image import rgb_to_lab


class ImageFileOrDirectory(Dataset):
    MODE_FILE = 'file'
    MODE_DIR = 'dir'

    DTYPE = np.float32

    def __init__(self, file_or_root, transform=None):
        if not os.path.exists(file_or_root):
            raise ValueError("'{}' does not exist".format(file_or_root))

        if os.path.isdir(file_or_root):
            self.mode = self.MODE_DIR
            self.root = file_or_root
            self.files = sorted(os.listdir(self.root))
        else:
            self.mode = self.MODE_FILE
            self.file = file_or_root

        self.transform = transform

    def __getitem__(self, index):
        if self.mode == self.MODE_DIR:
            path = os.path.join(self.root, self.files[index])
            return self._load_and_process_image(path), path
        elif self.mode == self.MODE_FILE:
            return self._load_and_process_image(self.file), self.file

    def __len__(self):
        if self.mode == self.MODE_DIR:
            return len(self.files)
        elif self.mode == self.MODE_FILE:
            return 1

    def _load_and_process_image(self, path):
        img = rgb_to_lab(io.imread(path)).astype(self.DTYPE)

        if self.transform:
            img = self.transform(img)

        return np.moveaxis(img, -1, 0)
