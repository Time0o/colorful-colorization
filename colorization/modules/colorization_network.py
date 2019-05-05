import os; os.environ['GLOG_minloglevel'] = '2'
from collections import OrderedDict
from functools import partial

import caffe
import torch; torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F

from ..cielab import ABGamut, CIELAB, DEFAULT_CIELAB
from .annealed_mean_decode_q import AnnealedMeanDecodeQ
from .conv2d_pad_same import Conv2dPadSame
from .get_class_weights import GetClassWeights
from .rebalance_loss import RebalanceLoss
from .soft_encode_ab import SoftEncodeAB


_CAFFE_LAYER_NAME_MAPPING = {
    'conv1_1': 'bw_conv1_1',
    'conv9_1': 'conv8_313'
}


class ColorizationNetwork(nn.Module):
    DEFAULT_KERNEL_SIZE = 3

    def __init__(self,
                 annealed_mean_T=0,
                 class_rebal_lambda=None,
                 device='cuda'):

        super().__init__()

        self.device = device

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

        self.conv9 = self._create_block(
            'conv9',
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
            self.conv9
        ]

        # en-/decoding
        self.encode_ab = SoftEncodeAB(DEFAULT_CIELAB,
                                      device=self.device)

        self.decode_q = AnnealedMeanDecodeQ(DEFAULT_CIELAB,
                                            T=annealed_mean_T,
                                            device=self.device)

        # rebalancing
        self.class_rebal_lambda = class_rebal_lambda

        if class_rebal_lambda is not None:
            self.get_class_weights = GetClassWeights(
                DEFAULT_CIELAB, lambda_=class_rebal_lambda)

            self.rebalance_loss = RebalanceLoss.apply

        # move to device
        self.to(self.device)

    def init_from_caffe(self, proto, model):
        # read in caffe network
        caffe_network = caffe.Net(proto, model, caffe.TEST)

        caffe_layers = {}

        for i, l in enumerate(caffe_network.layers[1:], start=1):
            layer_name = caffe_network._layer_names[i]

            if layer_name.startswith('relu'):
                continue

            caffe_layers[layer_name] = [b.data[...] for b in l.blobs]

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

            elif layer_type == 'batchnorm':
                norm = state_dict[param] = caffe_layers[layer][2]

                if norm == 0:
                    norm = 1

                if param_type == 'running_mean':
                    state_dict[param] = caffe_layers[layer][0] / norm
                elif param_type == 'running_var':
                    state_dict[param] = caffe_layers[layer][1] / norm

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
        l_norm = l - CIELAB.L_MEAN

        # prediction
        q_pred = l_norm
        for block in self._blocks:
            q_pred = block(q_pred)

        # label transformation
        if self.training:
            # downsample and encode labels
            ab = F.interpolate(ab, size=q_pred.shape[2:])
            q_actual = self.encode_ab(ab)

            # rebalancing
            if self.class_rebal_lambda is not None:
                color_weights = self.get_class_weights(q_actual)
                q_pred = self.rebalance_loss(q_pred, color_weights)

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

        # chain convolution layers
        block = nn.Sequential()

        for i in range(block_depth):
            layer = self._create_conv_layer(
                input_depth=(input_depth if i == 0 else output_depth),
                output_depth=output_depth,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                dilation=dilation,
                activation=activations)

            layer_name = '{}_{}'.format(name, i + 1)

            block.add_module(layer_name, layer)

        # optionally add batchnorm layer
        if batchnorm:
            bn = nn.Sequential(OrderedDict([
                ('batchnorm', nn.BatchNorm2d(output_depth, affine=False))
            ]))

            block.add_module('{}_{}norm'.format(name, block_depth), bn)

        return block

    def _create_conv_layer(self,
                           input_depth,
                           output_depth,
                           kernel_size,
                           stride,
                           dilation,
                           activation):

        layer = nn.Sequential()

        # convolution
        if stride < 1:
            _conv = partial(nn.ConvTranspose2d,
                            stride = int(1 / stride),
                            padding = (kernel_size - 1) // 2)
        else:
            if dilation > 1:
                _conv = partial(Conv2dPadSame,
                                stride = stride,
                                dilation=dilation)
            else:
                _conv = partial(nn.Conv2d,
                                stride = stride,
                                padding = (kernel_size - 1) // 2)

        conv = _conv(in_channels=input_depth,
                     out_channels=output_depth,
                     kernel_size=kernel_size)

        layer.add_module('conv', conv)

        # activation
        if activation:
            relu = nn.ReLU(inplace=True)

            layer.add_module('relu', relu)

        return layer
