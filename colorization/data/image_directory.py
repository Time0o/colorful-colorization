import os
from glob import glob
from mimetypes import types_map

from torch.utils.data.dataset import Dataset

from ..util.image import imread


class ImageDirectory(Dataset):
    LABEL_FILENAME = 'labels.txt'

    def __init__(self, root, transform=None, label_transform=int):
        self.root = root
        self.transform = transform
        self.label_transform = label_transform

        self._paths = []
        self._labels = None
        self._get_paths()

    def __getitem__(self, index):
        img = imread(self._paths[index])

        if self.transform:
            img = self.transform(img)

        if self._labels is not None:
            return img, self._labels[index]
        else:
            return img

    def __len__(self):
        return len(self._paths)

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
        image_extensions = self._image_extensions()

        for path in glob(os.path.join(self.root, '*')):
            ext = path.rsplit('.')[-1].lower()

            if any([ext == ext_ for ext_ in image_extensions]):
                self._paths.append(path)

        # check whether label file exists
        labels_path = os.path.join(self._root, self.LABEL_FILENAME)

        if not os.path.exists(labels_path):
            return

        # if so, build list of labels
        label_dict = {}

        with open(labels_path, 'r') as f:
            for line in f:
                path, val = line.split()

                path = os.path.join(self._root, path)
                val = self.label_transform(val)

                label_dict[path] = val

        self._labels = [label_dict[path] for path in self._paths]

    @staticmethod
    def _image_extensions():
        extensions = []
        for ext, t in types_map.items():
            if t.split('/')[0] == 'image':
                extensions.append(ext[1:])

        return extensions
