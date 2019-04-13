import os
from glob import glob

from skimage import io, color
from torch.utils.data.dataset import Dataset


class TinyImageNet(Dataset):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    def __init__(self, root, transform=None):
        self.set_root(root)
        self.set_dataset(self.TRAIN)
        self.set_labeled(True)

        self._build_indices()

    def __getitem__(self, index):
        image_path = self._indices[self.dataset][index]
        image_rgb = io.imread(image_path)
        image_lab = color.rgb2lab(image_rgb)

        if self.labeled:
            return image_lab[:, :, 0, np.newaxis], image_lab[:, :, 1:]
        else:
            return image_lab

    def __len__(self):
        return len(self._indices[self.dataset])

    def set_root(self, root):
        if not os.path.isdir(root):
            fmt = "not a directory: '{}'"
            raise ValueError(fmt.format(root))

        self.root = root

    def set_dataset(self, dataset):
        valid = [self.TRAIN, self.VAL, self.TEST]

        if dataset not in valid:
            fmt = "dataset must be either of {}"
            raise ValueError(fmt.format(', '.join(valid)))

        self.dataset = dataset

    def set_labeled(self, labeled):
        self.labeled = labeled

    def _build_indices(self):
        self._indices = {}

        for dataset in self.TRAIN, self.VAL, self.TEST:
            self._build_index(dataset)

    def _build_index(self, dataset):
        self._indices[dataset] = []

        dataset_path = os.path.join(self.root, dataset)

        if dataset == self.TRAIN:
            for images in self._listdir(dataset_path):
                images_root = os.path.join(images, 'images')

                for image_path in self._listdir(images_root):
                    self._indices[dataset].append(image_path)
        else:
            images_root = os.path.join(dataset_path, 'images')
            self._indices[dataset] = self._listdir(images_root)

    @staticmethod
    def _listdir(path):
        return sorted(glob(os.path.join(path, '*')))
