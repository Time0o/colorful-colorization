#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from colorization.cielab import ABGamut, CIELAB
from colorization.modules.decode_q import DecodeQ
from colorization.modules.encode_ab import EncodeAB
from colorization.test.util import resource_path


_SAMPLE_BATCH_SIZE = 32
_SAMPLE_HEIGHT = 100
_SAMPLE_WIDTH = 200

_AB_GAMUT_FILE = resource_path('ab-gamut.npy')
_AB_SAFE_RANGE = np.arange(0, 80, 10, dtype=np.float32)


class EncodeDecode(unittest.TestCase):
    def test_identity_mapping(self):
        # construct random (valid) ab values
        np.random.seed(0)

        ab_numpy = np.random.choice(
            _AB_SAFE_RANGE,
            size=(_SAMPLE_BATCH_SIZE, 2, _SAMPLE_HEIGHT, _SAMPLE_WIDTH))

        ab_torch = torch.from_numpy(ab_numpy)

        # encode and decode
        c = CIELAB(gamut=ABGamut.from_file(_AB_GAMUT_FILE))
        ab_dec = DecodeQ(c)(EncodeAB(c)(ab_torch))

        self.assertTrue(torch.all(ab_dec == ab_torch),
                        msg="decoding followed by encoding yields exact result")


if __name__ == '__main__':
    unittest.main()
