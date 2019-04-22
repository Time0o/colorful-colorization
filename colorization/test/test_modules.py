#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from colorization.cielab import ABGamut, CIELAB
from colorization.modules.cross_entropy_loss_2d import CrossEntropyLoss2d
from colorization.modules.decode_q import DecodeQ
from colorization.modules.encode_ab import EncodeAB
from colorization.test.util import resource_path


_SAMPLE_BATCH_SIZE = 32
_SAMPLE_HEIGHT = 100
_SAMPLE_WIDTH = 200

_AB_GAMUT_FILE = resource_path('ab-gamut.npy')
_AB_GAMUT = ABGamut.from_file(_AB_GAMUT_FILE)
_CIELAB = CIELAB(gamut=_AB_GAMUT)
_ENCODE = EncodeAB(_CIELAB)
_DECODE = DecodeQ(_CIELAB)

_AB_SAFE_RANGE = np.arange(0, 80, 10, dtype=np.float32)


def _random_ab(seed=0):
    np.random.seed(seed)

    ab_numpy = np.random.choice(
        _AB_SAFE_RANGE,
        size=(_SAMPLE_BATCH_SIZE, 2, _SAMPLE_HEIGHT, _SAMPLE_WIDTH))

    return torch.from_numpy(ab_numpy)


def _encode_ab(ab):
    return _ENCODE(ab)


def _decode_q(q):
    return _DECODE(q)


class EncodeABDecodeQCase(unittest.TestCase):
    def test_identity_mapping(self):
        # construct random (valid) ab values
        ab = _random_ab()

        # encode and decode
        ab_dec = _decode_q(_encode_ab(ab))

        self.assertTrue(torch.all(ab_dec == ab),
                        msg="decoding followed by encoding yields exact result")


class CrossEntropyLoss2dCase(unittest.TestCase):
    def test_zero_loss(self):
        # construct random (valid) ab values
        ab = _random_ab()

        # scale q so that softmax will return one hot values
        q = _encode_ab(ab) * 1000

        # check if loss is zero
        loss = CrossEntropyLoss2d()(q, q)

        self.assertEqual(0, loss, msg="cross entropy loss returns zero")

if __name__ == '__main__':
    unittest.main()
