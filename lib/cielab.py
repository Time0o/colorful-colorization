from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from skimage import color


class CIELAB:
    AB_BINSIZE = 10
    AB_RANGE = [-110, 110, AB_BINSIZE]
    AB_DTYPE = np.float64

    RGB_RESOLUTION = 101
    RGB_RANGE = [0, 1, RGB_RESOLUTION]
    RGB_DTYPE = np.float64

    def __init__(self, illuminant='D65', observer='2'):
        self.illuminant = illuminant
        self.observer = observer

        self._ab_grid = self._get_ab_grid()

        self._ab_gamut_mask = self._get_ab_gamut_mask(
            self._ab_grid, self.illuminant, self.observer)

        # TODO: assert number of True mask entries

    @classmethod
    def _get_ab_grid(cls):
        a = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)
        b = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)

        return np.dstack(np.meshgrid(a, b))

    @classmethod
    def _get_ab_gamut_mask(cls, ab_grid, illuminant, observer):
        # construct array of all points in discretized RGB space
        rgb_range = np.linspace(*cls.RGB_RANGE, dtype=cls.RGB_DTYPE)

        _rgb_space = np.meshgrid(rgb_range, rgb_range, rgb_range)
        rgb_space = np.stack(_rgb_space, -1).reshape(-1, 3)

        # convert points into Lab space
        ab_gamut = np.squeeze(color.rgb2lab(
            rgb_space[np.newaxis], illuminant, observer))[:, 1:]

        # find convex hull polygon of the resulting gamut
        ab_gamut_hull = ConvexHull(ab_gamut)
        ab_gamut_poly = Polygon(ab_gamut[ab_gamut_hull.vertices, :])

        # use polygon to construct "in-gamut" mask for discretized ab space
        ab_gamut_mask = np.full(ab_grid.shape[:-1], False, dtype=bool)

        for a in range(ab_grid.shape[0]):
            for b in range(ab_grid.shape[1]):
                for offs_a, offs_b in product([0, cls.AB_BINSIZE],
                                              [0, cls.AB_BINSIZE]):
                    a_, b_ = ab_grid[a, b]

                    if ab_gamut_poly.contains(Point(a_ + offs_a, b_ + offs_b)):
                        ab_gamut_mask[a, b] = True

        return ab_gamut_mask

    def plot_ab_gamut(self, l=50, ax=None):
        assert l >= 50 and l <= 100

        # construct Lab color space slice for given L
        l_ = np.full(self._ab_grid.shape[:2], l, dtype=self._ab_grid.dtype)
        color_space_lab = np.dstack((l_, self._ab_grid))

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
