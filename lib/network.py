import torch.nn as nn

from cielab import ABGamut


class ColorizationNetwork(nn.Module):
    KERNEL_SIZE = 3

    BETA1 = .9
    BETA2 = .99
    WEIGHT_DECAY = 1e-3
    LR_INIT = 3e-5

    def __init__(self):
        super().__init__()

        self.conv1 = self._create_block(
            'conv1', (2, 1, 64), strides=[1, 2])

        self.conv2 = self._create_block(
            'conv2', (2, 64, 128), strides=[1, 2])

        self.conv3 = self._create_block(
            'conv3', (3, 128, 256), strides=[1, 1, 2])

        self.conv4 = self._create_block(
            'conv4', (3, 256, 512), strides=[1, 1, 1])

        self.conv5 = self._create_block(
            'conv5', (3, 512, 512), strides=[1, 1, 1], dilation=2)

        self.conv6 = self._create_block(
            'conv6', (3, 512, 512), strides=[1, 1, 1], dilation=2)

        self.conv7 = self._create_block(
            'conv7', (3, 512, 256), strides=[1, 1, 1])

        self.conv8 = self._create_block(
            'conv8', (3, 256, 128), strides=[.5, 1, 1], batchnorm=False)

        self.conv9 = nn.Conv2d(in_channels=128,
                               out_channels=ABGamut.EXPECTED_SIZE,
                               kernel_size=1)

        self._blocks = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.conv7,
            self.conv8,
            self.conv9
        ]

    def forward(self, x):
        for block in self._blocks:
            x = block(x)

        return x

    @classmethod
    def _create_block(cls,
                      name,
                      dims,
                      strides,
                      dilation=1,
                      batchnorm=True):

        block_depth, input_depth, output_depth = dims

        # chain layers
        block = nn.Sequential()

        for i in range(block_depth):
            layer = cls._append_layer(
                input_depth=(input_depth if i == 0 else output_depth),
                output_depth=output_depth,
                stride=strides[i],
                dilation=dilation,
                batchnorm=(batchnorm and i == block_depth - 1))

            block.add_module('{}_{}'.format(name, i + 1), layer)

        return block

    @classmethod
    def _append_layer(cls,
                      input_depth,
                      output_depth,
                      stride=1,
                      dilation=1,
                      batchnorm=False):

        layer = nn.Sequential()

        # convolution
        conv = nn.Conv2d(in_channels=input_depth,
                         out_channels=output_depth,
                         kernel_size=cls.KERNEL_SIZE,
                         padding=(cls.KERNEL_SIZE - 1) // 2,
                         dilation=dilation)

        layer.add_module('conv', conv)

        # activation
        relu = nn.ReLU(inplace=True)

        layer.add_module('relu', relu)

        # batch normalization
        if batchnorm:
            bn = nn.BatchNorm2d(output_depth)

            layer.add_module('batchnorm', bn)

        return layer
