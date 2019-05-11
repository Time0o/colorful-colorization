from collections import OrderedDict

import torch.nn as nn


class Conv2dSeparable(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 relu_first=False):

        super().__init__()

        self.relu_first = relu_first

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               padding=dilation,
                               stride=stride,
                               dilation=dilation,
                               groups=in_channels,
                               bias=False)),
            ('bn', nn.BatchNorm2d(in_channels))
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_channels))
        ]))

        self.relu1 = nn.ReLU()

        if not relu_first:
            self.relu2 = nn.ReLU()

    def forward(self, x):
        if self.relu_first:
            x = self.relu1(x)
            x = self.conv1(x)
            x = self.conv2(x)
        else:
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)

        return x
