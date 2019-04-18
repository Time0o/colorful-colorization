import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cielab import ABGamut
from encode_ab import EncodeAB
from interpolate import Interpolate
from loss import CrossEntropyLoss2d


class ColorizationNetwork(nn.Module):
    KERNEL_SIZE = 3

    BETA1 = .9
    BETA2 = .99
    WEIGHT_DECAY = 1e-3
    LR_INIT = 3e-5

    def __init__(self, input_size, cielab):
        super().__init__()

        # prediction
        size = input_size

        self.conv1, size = self._create_block(
            'conv1', (2, size, 1, 64), strides=[1, 2])

        self.conv2, size = self._create_block(
            'conv2', (2, size, 64, 128), strides=[1, 2])

        self.conv3, size = self._create_block(
            'conv3', (3, size, 128, 256), strides=[1, 1, 2])

        self.conv4, size = self._create_block(
            'conv4', (3, size, 256, 512), strides=[1, 1, 1])

        self.conv5, size = self._create_block(
            'conv5', (3, size, 512, 512), strides=[1, 1, 1], dilation=2)

        self.conv6, size = self._create_block(
            'conv6', (3, size, 512, 512), strides=[1, 1, 1], dilation=2)

        self.conv7, size = self._create_block(
            'conv7', (3, size, 512, 256), strides=[1, 1, 1])

        self.conv8, size = self._create_block(
            'conv8', (3, size, 256, 128), strides=[.5, 1, 1], batchnorm=False)

        self.classify = nn.Conv2d(in_channels=128,
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
            self.classify
        ]

        # label transformation
        self.downsample =  Interpolate(size / input_size)

        self.encode_ab = EncodeAB(cielab)

    def forward(self, img):
        l, ab = img[:, :1, :, :], img[:, 1:, :, :]

        # prediction
        q_pred = l
        for block in self._blocks:
            if self._is_dilating_block(block):
                torch.backends.cudnn.benchmark = True
                q_pred = block(q_pred)
                torch.backends.cudnn.benchmark = False
            else:
                q_pred = block(q_pred)

        # label transformation
        ab = self.downsample(ab)

        q_actual = self.encode_ab(ab)

        return q_pred, q_actual

    def run_training(self, dataloader, iterations, device=None, verbosity=0):
        # validate dataset properties
        dataset = dataloader.dataset

        assert dataset.color_space == dataset.COLOR_SPACE_LAB

        # switch to training mode (essential for batch normalization)
        was_training = self.training

        if not was_training:
            self.train()

        # move model to device
        if device is not None:
            self.to(device)

        # create optimizer
        op = optim.Adam(self.parameters(),
                        lr=self.LR_INIT,
                        betas=(self.BETA1, self.BETA2),
                        weight_decay=self.WEIGHT_DECAY)

        # optimization loop
        criterion = CrossEntropyLoss2d()

        i = 1
        while i <= iterations:
            for img in dataloader:
                # move data to device
                if device is not None:
                    img = img.to(device)

                # perform parameter update
                op.zero_grad()

                q_pred, q_actual = self(img)
                loss = criterion(q_pred, q_actual)
                loss.backward()

                op.step()

                if verbosity > 0:
                    fmt = "iteration {:,}/{:,}: loss was {:1.3e}"
                    msg = fmt.format(i, iterations, loss)

                    if verbosity == 1:
                        end = '\n' if i == iterations else ''
                        msg = '\r' + msg.ljust(50)
                    elif verbosity > 1:
                        end = '\n'

                    print(msg, end=end, flush=True)

                i += 1

                if i > iterations:
                    break

        # reset model mode
        if not was_training:
            self.eval()

    @classmethod
    def _create_block(cls,
                      name,
                      dims,
                      strides,
                      dilation=1,
                      batchnorm=True):

        block_depth, input_size, input_depth, output_depth = dims

        # chain layers
        block = nn.Sequential()

        for i in range(block_depth):
            layer = cls._append_layer(
                input_size=input_size,
                input_depth=(input_depth if i == 0 else output_depth),
                output_depth=output_depth,
                stride=strides[i],
                dilation=dilation,
                batchnorm=(batchnorm and i == block_depth - 1))

            block.add_module('{}_{}'.format(name, i + 1), layer)

            # update input size based on stride
            input_size /= strides[i]

        return block, input_size

    @classmethod
    def _append_layer(cls,
                      input_size,
                      input_depth,
                      output_depth,
                      stride=1,
                      dilation=1,
                      batchnorm=False):

        layer = nn.Sequential()

        # upsampling
        if stride < 1:
            upsample = Interpolate(1 / stride)

            layer.add_module('upsample', upsample)

            stride = 1

        # adjust padding for dilated convolutions
        if dilation > 1:
            padding = cls._dilation_padding(
                i=input_size,
                k=cls.KERNEL_SIZE,
                s=stride,
                d=dilation)
        else:
            padding = (cls.KERNEL_SIZE - 1) // 2

        # convolution
        conv = nn.Conv2d(in_channels=input_depth,
                         out_channels=output_depth,
                         kernel_size=cls.KERNEL_SIZE,
                         stride=stride,
                         padding=padding,
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

    @staticmethod
    def _tensor_GiB(t):
        return t.element_size() * t.nelement() / 1024**3

    @staticmethod
    def _dilation_padding(i, k, s, d):
        # calculates necessary padding to preserve input shape when applying
        # dilated convolution, unlike Keras, PyTorch does not provide a way to
        # calculate this automatidally
        return int(((i - 1) * (s - 1) + d * (k - 1)) / 2)

    @staticmethod
    def _is_dilating_block(block):
        # returns True if any of the layers in a block use dilated convolutions,
        # in that case we have to temporarily set torch.backends.cudnn.benchmark
        # to True because a forward pass through the those layers will otherwise
        # be extermely slow due to a bug in PyTorch
        for layer in list(block.modules())[1:]:
            if isinstance(layer, nn.Conv2d) and layer.dilation != (1, 1):
                return True

        return False
