import colour
import matplotlib.pyplot as plt
import numpy as np
import skimage


class CIELAB:
    BIN_SIZE = 10

    RANGE_A = [-110, 110, BIN_SIZE]
    RANGE_B = [-110, 110, BIN_SIZE]

    @classmethod
    def color_space(cls, l=50, gamut_only=True, ax=None):
        assert l >= 50

        if ax is None:
            _, ax = plt.subplots()

        # construct color space
        a = np.arange(*cls.RANGE_A, dtype=float)
        b = np.arange(*cls.RANGE_B, dtype=float)

        b_, a_ = np.meshgrid(b, a)

        l_ = np.full_like(a_, l)

        color_space_lab = np.dstack((l_, a_, b_))
        color_space_rgb = skimage.color.lab2rgb(color_space_lab)

        # remove out of gamut colors
        if gamut_only:
            mask = colour.is_within_visible_spectrum(
                colour.Lab_to_XYZ(color_space_lab))

            color_space_rgb[~mask, :] = 1

        # display color space
        ax.imshow(color_space_rgb,
                  extent=[*cls.RANGE_B[:2], *cls.RANGE_A[1::-1]])

        # set axis labels and title
        ax.set_xlabel("$b$")
        ax.set_ylabel("$a$")

        ax.set_title(r"$RGB(a, b \mid L = {})$".format(l))

        # customize ticks and grid
        ax.set_xticks(np.linspace(*cls.RANGE_B[:2], 5))
        ax.set_yticks(np.linspace(*cls.RANGE_A[:2], 5))

        ax.grid(color='k', linestyle=':', dashes=(1, 4))
