from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from skimage import color


class ABGamut:
    EXPECTED_SIZE = 313

    def __init__(self, points=None):
        self.points = points

    @classmethod
    def auto(cls):
        return cls()

    @classmethod
    def from_file(cls, file):
        points = np.load(file)

        assert points.shape[0] == cls.EXPECTED_SIZE
        assert points.shape[1] == 2

        return cls(points=points)


class CIELAB:
    AB_BINSIZE = 10
    AB_RANGE = [-110, 110, AB_BINSIZE]
    AB_DTYPE = np.float64

    RGB_RESOLUTION = 101
    RGB_RANGE = [0, 1, RGB_RESOLUTION]
    RGB_DTYPE = np.float64

    L_MEAN = 50
    L_STD = 50
    Q_DTYPE = np.int64

    def __init__(self, gamut=ABGamut.auto()):
        self.gamut = gamut

        a, b, self.ab = self._get_ab()

        self.ab_gamut_mask = self._get_ab_gamut_mask(
            a, b, self.ab, self.gamut)

        self.ab_to_q = self._get_ab_to_q(self.ab_gamut_mask)

        self.q_to_ab = self._get_q_to_ab(self.ab, self.ab_gamut_mask)

    @classmethod
    def _get_ab(cls):
        a = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)
        b = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)

        b_, a_ = np.meshgrid(a, b)
        ab = np.dstack((a_, b_))

        return a, b, ab

    @classmethod
    def _get_ab_gamut_mask(cls, a, b, ab, gamut):
        ab_gamut_mask = np.full(ab.shape[:-1], False, dtype=bool)

        if gamut.points is not None:
            a = np.digitize(gamut.points[:, 0], a) - 1
            b = np.digitize(gamut.points[:, 1], b) - 1

            for a_, b_ in zip(a, b):
                ab_gamut_mask[a_, b_] = True
        else:
            # construct array of all points in discretized RGB space
            rgb_range = np.linspace(*cls.RGB_RANGE, dtype=cls.RGB_DTYPE)

            _rgb_space = np.meshgrid(rgb_range, rgb_range, rgb_range)
            rgb_space = np.stack(_rgb_space, -1).reshape(-1, 3)

            # convert points into Lab space
            ab_gamut = np.squeeze(color.rgb2lab(rgb_space[np.newaxis]))[:, 1:]

            # find convex hull polygon of the resulting gamut
            ab_gamut_hull = ConvexHull(ab_gamut)
            ab_gamut_poly = Polygon(ab_gamut[ab_gamut_hull.vertices, :])

            # use polygon to construct "in-gamut" mask for discretized ab space
            for a in range(ab.shape[0]):
                for b in range(ab.shape[1]):
                    for offs_a, offs_b in product([0, cls.AB_BINSIZE],
                                                  [0, cls.AB_BINSIZE]):
                        a_, b_ = ab[a, b]

                        if ab_gamut_poly.contains(Point(a_ + offs_a, b_ + offs_b)):
                            ab_gamut_mask[a, b] = True

        return ab_gamut_mask

    @classmethod
    def _get_ab_to_q(cls, ab_gamut_mask):
        ab_to_q = np.full(ab_gamut_mask.shape, -1, dtype=cls.Q_DTYPE)

        ab_to_q[ab_gamut_mask] = np.arange(np.count_nonzero(ab_gamut_mask))

        return ab_to_q

    @classmethod
    def _get_q_to_ab(cls, ab, ab_gamut_mask):
        return ab[ab_gamut_mask] + cls.AB_BINSIZE / 2

    @classmethod
    def _plot_ab_matrix(cls, mat, ax=None, title=None):
        if ax is None:
            _, ax = plt.subplots()

        imshow = partial(ax.imshow,
                         np.flip(mat, axis=0),
                         extent=[*cls.AB_RANGE[:2]] * 2)

        if len(mat.shape) < 3 or mat.shape[2] == 1:
            im = imshow(cmap='jet')

            fig = plt.gcf()
            fig.colorbar(im, cax=fig.add_axes())
        else:
            imshow()

        # set title
        if title is not None:
            ax.set_title(title)

        # set axes labels
        ax.set_xlabel("$b$")
        ax.set_ylabel("$a$")

        # set ticks
        ax.set_xticks(np.linspace(*cls.AB_RANGE[:2], 5))
        ax.set_yticks(np.linspace(*cls.AB_RANGE[:2], 5))
        ax.invert_yaxis()

        # set grid
        ax.grid(color='k', linestyle=':', dashes=(1, 4))

    def plot_ab_gamut(self, l=50, ax=None):
        assert l >= 50 and l <= 100

        # construct Lab color space slice for given L
        l_ = np.full(self.ab.shape[:2], l, dtype=self.ab.dtype)
        color_space_lab = np.dstack((l_, self.ab))

        # convert to RGB
        color_space_rgb = color.lab2rgb(color_space_lab)

        # mask out of gamut colors
        color_space_rgb[~self.ab_gamut_mask, :] = 1

        # display color space
        self._plot_ab_matrix(
            color_space_rgb, ax, r"$RGB(a, b \mid L = {})$".format(l))

    def plot_empirical_distribution(self, dataset, ax=None, verbose=False):
        # accumulate ab values
        ab_acc = np.zeros([self.AB_RANGE[1] - self.AB_RANGE[0]] * 2)

        for i, img in enumerate(dataset):
            if verbose:
                fmt = "\rprocessing image {}/{}"

                print(fmt.format(i + 1, len(dataset)),
                      end=('\n' if i == len(dataset) - 1 else ''),
                      flush=True)

            img = np.moveaxis(img, 0, -1)
            ab_rounded = np.round(img[:, :, 1:].reshape(-1, 2)).astype(int)
            ab_offset = ab_rounded - self.AB_RANGE[0]

            np.add.at(ab_acc, tuple(np.split(ab_offset, 2, axis=1)), 1)

        # convert to log scale
        ab_acc[ab_acc == 0] = np.nan

        ab_acc_log = np.log10(ab_acc) - np.log10(len(dataset))

        # display distribution
        self._plot_ab_matrix(ab_acc_log, ax, r"$log(P(a, b))$")
