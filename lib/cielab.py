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

        return cls(points=np.flip(points, axis=1))


class CIELAB:
    AB_BINSIZE = 10
    AB_RANGE = [-110, 110, AB_BINSIZE]
    AB_DTYPE = np.float64

    RGB_RESOLUTION = 101
    RGB_RANGE = [0, 1, RGB_RESOLUTION]
    RGB_DTYPE = np.float64

    def __init__(self, gamut=ABGamut.auto()):
        self._a, self._b, self._ab = self._get_ab()

        self._ab_gamut_mask = self._get_ab_gamut_mask(
            self._a, self._b, self._ab, gamut)

    @classmethod
    def _get_ab(cls):
        a = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)
        b = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)

        return a, b, np.dstack(np.meshgrid(a, b))

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

    def plot_ab_gamut(self, l=50, ax=None):
        assert l >= 50 and l <= 100

        # construct Lab color space slice for given L
        l_ = np.full(self._ab.shape[:2], l, dtype=self._ab.dtype)
        color_space_lab = np.dstack((l_, self._ab))

        # convert to RGB
        color_space_rgb = color.lab2rgb(color_space_lab)

        # mask out of gamut colors
        color_space_rgb[~self._ab_gamut_mask, :] = 1

        # display color space
        color_space_rgb = np.flip(color_space_rgb.transpose(1, 0, 2), axis=0)

        if ax is None:
            _, ax = plt.subplots()

        ax.imshow(color_space_rgb, extent=[*self.AB_RANGE[:2]] * 2)

        # set axis labels and title
        ax.set_xlabel("$b$")
        ax.set_ylabel("$a$")

        ax.set_title(r"$RGB(a, b \mid L = {})$".format(l))

        # customize ticks and grid
        ax.set_xticks(np.linspace(*self.AB_RANGE[:2], 5))
        ax.set_yticks(np.linspace(*self.AB_RANGE[:2], 5))
        ax.invert_yaxis()

        ax.grid(color='k', linestyle=':', dashes=(1, 4))
