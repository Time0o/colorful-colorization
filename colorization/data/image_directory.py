import os
from glob import glob
from mimetypes import types_map

from torch.utils.data.dataset import Dataset

from ..util.image import imread


class ImageDirectory(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self._paths = []
        self._get_paths()

    def __getitem__(self, index):
        img = imread(self._paths[index])

        if self.transform:
            img = self.transform(img)

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
        image_extensions = self._image_extensions()

        for path in glob(os.path.join(self.root, '*')):
            ext = path.rsplit('.')[-1].lower()

            if any([ext == ext_ for ext_ in image_extensions]):
                self._paths.append(path)

    @staticmethod
    def _image_extensions():
        extensions = []
        for ext, t in types_map.items():
            if t.split('/')[0] == 'image':
                extensions.append(ext[1:])

        return extensions
