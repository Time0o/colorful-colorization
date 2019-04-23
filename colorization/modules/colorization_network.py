import os
from collections import OrderedDict

try:
    os.environ['GLOG_minloglevel'] = '2'

    import caffe

    _caffe_available = True

    _CAFFE_LAYER_NAME_MAPPING = {
        'conv1_1': 'bw_conv1_1',
        'classify': 'conv8_313'
    }

    _CAFFE_LAYERS_SUPERFLUOUS = {
        'conv8_313_rh', 'class8_313_rh', 'class8_ab', 'Silence'
    }

except ImportError:
    _caffe_available = False

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cielab import ABGamut, DEFAULT_CIELAB
from .conv2d_pad_same import Conv2dPadSame
from .cross_entropy_loss_2d import CrossEntropyLoss2d
from .decode_q import DecodeQ
from .encode_ab import EncodeAB


class ColorizationNetwork(nn.Module):
    DEFAULT_KERNEL_SIZE = 3

    def __init__(self):
        super().__init__()

        DEFAULT_CIELAB

        # prediction
        self.conv1 = self._create_block('conv1', (2, 1, 64), strides=[1, 2])
        self.conv2 = self._create_block('conv2', (2, 64, 128), strides=[1, 2])
        self.conv3 = self._create_block('conv3', (3, 128, 256), strides=[1, 1, 2])
        self.conv4 = self._create_block('conv4', (3, 256, 512))
        self.conv5 = self._create_block('conv5', (3, 512, 512), dilation=2)
        self.conv6 = self._create_block('conv6', (3, 512, 512), dilation=2)
        self.conv7 = self._create_block('conv7', (3, 512, 512))

        self.conv8 = self._create_block(
            'conv8',
            (3, 512, 256),
            kernel_sizes=[4, 3, 3],
            strides=[.5, 1, 1],
            batchnorm=False
        )

        self.classify = self._create_block(
            'classify',
            (1, 256, ABGamut.EXPECTED_SIZE),
            kernel_sizes=[1],
            strides=[1],
            batchnorm=False,
            activations=False
        )

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
        self.encode_ab = EncodeAB(DEFAULT_CIELAB)
        self.decode_q = DecodeQ(DEFAULT_CIELAB)

    def init_from_caffe(self, proto, model):
        if not _caffe_available:
            raise ValueError("caffe not available, can not read caffe model")

        # read in caffe network
        caffe_network = caffe.Net(proto, model, caffe.TEST)

        caffe_layers = {}

        for i, l in enumerate(caffe_network.layers[1:], start=1):
            layer_name = caffe_network._layer_names[i]

            if layer_name.startswith('relu'):
                continue

            layer_weights = [b.data[...] for b in l.blobs]

            caffe_layers[layer_name] = layer_weights

        caffe_layers_unused = set(caffe_layers)

        # create pytorch state dict from pretrained caffe weights
        state_dict = {}

        for param in self.state_dict():
            layer, layer_type, param_type = param.split('.', 1)[1].split('.')

            layer = _CAFFE_LAYER_NAME_MAPPING.get(layer, layer)

            if layer_type == 'conv':
                if param_type == 'weight':
                    state_dict[param] = caffe_layers[layer][0]
                elif param_type == 'bias':
                    state_dict[param] = caffe_layers[layer][1]
                else:
                    assert False

            elif layer_type == 'batchnorm':
                norm = state_dict[param] = caffe_layers[layer][2]

                if param_type == 'running_mean':
                    state_dict[param] = caffe_layers[layer][0] / norm
                elif param_type == 'running_var':
                    state_dict[param] = caffe_layers[layer][1] / norm

            else:
                raise ValueError("unhandled layer type '{}'".format(layer_type))

            if layer in caffe_layers_unused:
                caffe_layers_unused.remove(layer)

        # assert that no caffe parameter remain unused
        caffe_layers_unused -= _CAFFE_LAYERS_SUPERFLUOUS

        if caffe_layers_unused:
            fmt = "unused caffe layers: {}"
            raise ValueError(fmt.format(', '.join(sorted(caffe_layers_unused))))

        # load state dict
        self.load_state_dict({
            k: torch.from_numpy(v) for k, v in state_dict.items()
        })

    def forward(self, img):
        if self.training:
            l, ab = img[:, :1, :, :], img[:, 1:, :, :]
        else:
            l = img

        # normalize lightness
        l_norm = l - DEFAULT_CIELAB.L_MEAN

        # prediction
        q_pred = l_norm
        for block in self._blocks:
            if self._is_dilating_block(block):
                torch.backends.cudnn.benchmark = True
                q_pred = block(q_pred)
                torch.backends.cudnn.benchmark = False
            else:
                q_pred = block(q_pred)

        # label transformation
        if self.training:
            ab = F.interpolate(ab, size=q_pred.shape[2:])

            q_actual = self.encode_ab(ab)

            return q_pred, q_actual
        else:
            ab_pred = self.decode_q(q_pred)

            return ab_pred

    def _create_block(self,
                      name,
                      dims,
                      kernel_sizes=None,
                      strides=None,
                      dilation=1,
                      batchnorm=True,
                      activations=True):

        # block dimensions
        block_depth, input_depth, output_depth = dims

        # default kernel sizes and strides
        if strides is None:
            strides = [1] * block_depth

        if kernel_sizes is None:
            kernel_sizes = [self.DEFAULT_KERNEL_SIZE] * block_depth

        # chain layers
        block = nn.Sequential()

        for i in range(block_depth):
            layer = self._append_layer(
                input_depth=(input_depth if i == 0 else output_depth),
                output_depth=output_depth,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                dilation=dilation,
                activation=activations)

            if block_depth == 1:
                layer_name = name
            else:
                layer_name = '{}_{}'.format(name, i + 1)

            block.add_module(layer_name, layer)

        # optionally add batchnorm layer
        if batchnorm:
            bn = nn.Sequential(OrderedDict([
                ('batchnorm', nn.BatchNorm2d(output_depth, affine=False))
            ]))

            block.add_module('{}_{}norm'.format(name, block_depth), bn)

        return block

    def _append_layer(self,
                      input_depth,
                      output_depth,
                      kernel_size,
                      stride,
                      dilation,
                      activation):

        layer = nn.Sequential()

        # convolution
        if stride < 1:
            conv = nn.ConvTranspose2d(in_channels=input_depth,
                                      out_channels=output_depth,
                                      kernel_size=kernel_size,
                                      stride=int(1 / stride),
                                      padding=(kernel_size - 1) // 2)
        else:
            conv = Conv2dPadSame(in_channels=input_depth,
                                 out_channels=output_depth,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 dilation=dilation)

        layer.add_module('conv', conv)

        # activation
        if activation:
            relu = nn.ReLU(inplace=True)

            layer.add_module('relu', relu)

        return layer

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
