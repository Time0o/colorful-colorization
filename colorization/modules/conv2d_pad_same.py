from math import ceil

import torch
import torch.nn as nn


class Conv2dPadSame(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1):

        assert isinstance(kernel_size, int)
        assert isinstance(stride, int)
        assert isinstance(dilation, int)

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         dilation=dilation)

        self.k = kernel_size
        self.s = stride
        self.d = dilation

    def forward(self, x):
        n, c, h, w = x.shape

        h_out = ceil(h / self.s)
        w_out = ceil(w / self.s)

        pad_rows = (h_out - 1) * self.s + (self.k - 1) * self.d + 1 - h
        pad_cols = (w_out - 1) * self.s + (self.k - 1) * self.d + 1 - w

        pad_rows1 = pad_rows // 2
        pad_rows2 = pad_rows - pad_rows1

        pad_cols1 = pad_cols // 2
        pad_cols2 = pad_cols - pad_cols1

        x_padded = nn.ZeroPad2d((pad_rows1, pad_rows2, pad_cols1, pad_cols2))(x)

        return super().forward(x_padded)
