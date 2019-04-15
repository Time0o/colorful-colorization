import os
from glob import glob

from skimage import io, color
from torch.utils.data.dataset import Dataset

from cielab import CIELAB


class TinyImageNet(Dataset):
    DATASET_TRAIN = 'train'
    DATASET_VAL = 'val'
    DATASET_TEST = 'test'

    COLOR_SPACE_RGB = 'rgb'
    COLOR_SPACE_LAB = 'lab'

    def __init__(self,
                 root,
                 dataset=DATASET_TRAIN,
                 labeled=True,
                 cielab=CIELAB(),
                 color_space=COLOR_SPACE_LAB,
                 transform=None):

        self.set_root(root)
        self.set_dataset(dataset)
        self.set_labeled(labeled)
        self.set_cielab(cielab)
        self.set_color_space(color_space)

        self._build_indices()

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

    def set_labeled(self, labeled):
        self.labeled = labeled

    def set_cielab(self, cielab):
        self.cielab = cielab

    def set_color_space(self, color_space):
        if color_space not in [self.COLOR_SPACE_RGB, self.COLOR_SPACE_LAB]:
            raise ValueError("invalid color space")

        self.color_space = color_space

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

                for image_path in self._listdir(images_root):
                    self._indices[dataset].append(image_path)
        else:
            images_root = os.path.join(dataset_path, 'images')
            self._indices[dataset] = self._listdir(images_root)

    def _getitem(self, index):
        image_path = self._indices[self.dataset][index]
        image_rgb = io.imread(image_path)

        if self.color_space == self.COLOR_SPACE_RGB:
            if self.labeled:
                raise ValueError("can not produce labeled data from RGB images")

            return image_rgb

        elif self.color_space == self.COLOR_SPACE_LAB:
            image_lab = self.cielab.rgb_to_lab(image_rgb)

            if self.labeled:
                return self.cielab.dissemble(image_lab)
            else:
                return image_lab

    @staticmethod
    def _listdir(path):
        return sorted(glob(os.path.join(path, '*')))
