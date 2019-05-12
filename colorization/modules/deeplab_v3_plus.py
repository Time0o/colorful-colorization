from collections import OrderedDict
from functools import partial
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2d_separable import Conv2dSeparable


def _conv1x1(in_channels, out_channels):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU())
    ]))


def _interpolate(x, to):
    return F.interpolate(x, size=to, mode='bilinear', align_corners=True)


class _XceptionBlock(nn.Module):
    def __init__(self,
                 channels,
                 stride=1,
                 dilation=1,
                 skip_type='conv',
                 ll=False,
                 relu_first=True):

        super().__init__()

        self.skip_type = skip_type
        self.sum = sum
        self.ll = ll

        if skip_type == 'conv':
            self.conv0 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    channels[0], channels[-1], 1, stride=stride, bias=False)),
                ('bn', nn.BatchNorm2d(channels[-1]))
            ]))

        conv_separable = partial(Conv2dSeparable,
                                 kernel_size=3,
                                 dilation=dilation,
                                 relu_first=relu_first)

        self.conv1 = conv_separable(channels[0], channels[1])
        self.conv2 = conv_separable(channels[1], channels[2])
        self.conv3 = conv_separable(channels[2], channels[3], stride=stride)

    def forward(self, x):
        ll = self.conv2(self.conv1(x))
        res = self.conv3(ll)

        if self.skip_type == 'conv':
            x = res + self.conv0(x)
        elif self.skip_type == 'sum':
            x = res + x
        else:
            x = res

        if self.ll:
            return ll, x
        else:
            return x


class _Xception(nn.Module):
    ENTRY_STRIDE = 2
    MAIN_DILATION = 1
    EXIT_DILATIONS = (1, 2)

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU())
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU())
        ]))

        self.entry_block1 = _XceptionBlock([64, 128, 128, 128],
                                           stride=2)

        self.entry_block2 = _XceptionBlock([128, 256, 256, 256],
                                           stride=2,
                                           ll=True)

        self.entry_block3 = _XceptionBlock([256, 728, 728, 728],
                                           stride=self.ENTRY_STRIDE)

        self.main_blocks = nn.Sequential(OrderedDict([
            (
                'block{}'.format(i + 1),
                _XceptionBlock(
                    [728] * 4, dilation=self.MAIN_DILATION, skip_type='sum')
            )
            for i in range(16)
        ]))

        self.exit_block1 = _XceptionBlock([728, 728, 1024, 1024],
                                          dilation=self.EXIT_DILATIONS[0])

        self.exit_block2 = _XceptionBlock([1024, 1536, 1536, 2048],
                                          dilation=self.EXIT_DILATIONS[1],
                                          skip_type='none',
                                          relu_first=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.entry_block1(x)
        ll, x = self.entry_block2(x)
        x = self.entry_block3(x)

        x = self.main_blocks(x)

        x = self.exit_block1(x)
        x = self.exit_block2(x)

        return ll, x


class _AtrousSpatialPyramidPooling(nn.Module):
    DILATIONS = [6, 12, 18]
    DROPOUT_PROB = 0.1

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_conv = _conv1x1(in_channels, out_channels)

        # parallel convolution layers
        self.conv_parallel1 = _conv1x1(in_channels, out_channels)

        def conv_separable(dilation):
            return Conv2dSeparable(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   dilation=dilation)

        self.conv_parallel2 = conv_separable(self.DILATIONS[0])
        self.conv_parallel3 = conv_separable(self.DILATIONS[1])
        self.conv_parallel4 = conv_separable(self.DILATIONS[2])

        # final convolution layer
        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(5 * out_channels, out_channels, 1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout2d(p=self.DROPOUT_PROB))
        ]))

    def forward(self, x):
        pool = self.pool(x)
        pool = self.pool_conv(pool)
        pool = _interpolate(pool, x.shape[2:])

        x = torch.cat((
            pool,
            self.conv_parallel1(x),
            self.conv_parallel2(x),
            self.conv_parallel3(x),
            self.conv_parallel4(x)
        ), dim=1)

        x = self.conv(x)

        return x


class _Decoder(nn.Module):
    def __init__(self,
                 ll_in_channels,
                 ll_reduced_channels,
                 enc_in_channels,
                 enc_out_channels):

        super().__init__()

        self.ll_conv = _conv1x1(ll_in_channels, ll_reduced_channels)

        self.conv1 = Conv2dSeparable(
            in_channels=(enc_in_channels + ll_reduced_channels),
            out_channels=enc_in_channels,
            kernel_size=3)

        self.conv2 = Conv2dSeparable(
            in_channels=enc_in_channels,
            out_channels=enc_out_channels,
            kernel_size=3)

    def forward(self, ll, enc):
        enc = _interpolate(enc, ll.shape[2:])

        ll = self.ll_conv(ll)

        comb = torch.cat((enc, ll), dim=1)
        comb = self.conv1(comb)
        comb = self.conv2(comb)

        return comb


class _TFConverter:
    def __init__(self, tf, checkpoint):
        self.tf_reader = tf.train.NewCheckpointReader(checkpoint)
        self.prefix = None

        self._processed_tensors = set()

    def set_prefix(self, prefix):
        self.prefix = prefix

    def warn_if_incomplete(self, ignore_logits=False):
        all_tensors = set()

        for tensor in self.tf_reader.get_variable_to_shape_map():
            if not self._ignore_tensor(tensor, ignore_logits=ignore_logits):
                all_tensors.add(tensor)

        if self._processed_tensors != all_tensors:
            not_processed = sorted(all_tensors - self._processed_tensors)

            fmt = "the following tensors were not processed:\n{}"
            warn(fmt.format('\n'.join(not_processed)))

    def conv(self, pt_layer, tf_layer, separable=False, bias=False):
        if separable:
            self.conv(pt_layer.conv1, tf_layer + '_depthwise')
            self.conv(pt_layer.conv2, tf_layer + '_pointwise')
        else:
            if tf_layer.endswith('_depthwise'):
                pt_layer.conv.weight.data = self._get(
                    pt_layer.conv.weight.data,
                    tf_layer,
                    'depthwise_weights',
                    transpose=(2, 3, 0, 1))
            else:
                pt_layer.conv.weight.data = self._get(
                    pt_layer.conv.weight.data,
                    tf_layer,
                    'weights',
                    transpose=(3, 2, 0, 1))

            if bias:
                pt_layer.conv.bias.data = self._get(
                    pt_layer.conv.bias.data, tf_layer, 'biases')

            if hasattr(pt_layer, 'bn'):
                self.batchnorm(pt_layer.bn, tf_layer + '/BatchNorm')

    def batchnorm(self, pt_layer, tf_layer):
        pt_layer.bias.data = self._get(
            pt_layer.bias.data, tf_layer, 'beta')
        pt_layer.weight.data = self._get(
            pt_layer.weight.data, tf_layer, 'gamma')
        pt_layer.running_mean.data = self._get(
            pt_layer.running_mean.data, tf_layer, 'moving_mean')
        pt_layer.running_var.data = self._get(
            pt_layer.running_var.data, tf_layer, 'moving_variance')

    def xception_block(self, pt_block, tf_block):
        if pt_block.skip_type == 'conv':
            self.conv(pt_block.conv0, tf_block + '/shortcut')

        self.conv(pt_block.conv1, tf_block + '/separable_conv1', separable=True)
        self.conv(pt_block.conv2, tf_block + '/separable_conv2', separable=True)
        self.conv(pt_block.conv3, tf_block + '/separable_conv3', separable=True)

    def _get(self, pt_tensor, tf_layer, tf_tensor, transpose=None):
        # load tensor
        tf_path = "{}/{}".format(tf_layer, tf_tensor)
        if self.prefix is not None:
            tf_path = self.prefix + tf_path

        tf_tensor = self.tf_reader.get_tensor(tf_path)

        # transpose tensor if necessary
        if transpose is not None:
            tf_tensor = tf_tensor.transpose(transpose)

        # assert that tensor shapes match
        if tuple(pt_tensor.shape) != tf_tensor.shape:
            raise ValueError("tensor shape mismatch for '{}'".format(tf_path))

        # mark tensor as processed
        self._processed_tensors.add(tf_path)

        return torch.Tensor(tf_tensor)

    @staticmethod
    def _ignore_tensor(tensor, ignore_logits=False):
        keywords = ['global_step']
        postfixes = ['Momentum', 'ExponentialMovingAverage']

        if ignore_logits and 'logits' in tensor:
            return True

        if any([(k in tensor) for k in keywords]):
            return True

        if any([tensor.endswith(p) for p in postfixes]):
            return True


class DeepLabV3Plus(nn.Module):
    ASPP_IN_CHANNELS = 2048
    ASPP_OUT_CHANNELS = 256

    DECODER_IN_CHANNELS = 256
    DECODER_REDUCED_CHANNELS = 48
    DECODER_OUT_CHANNELS = 256

    def __init__(self, out_channels):
        super().__init__()

        self.encoder = _Xception()

        self.aspp = _AtrousSpatialPyramidPooling(
            in_channels=self.ASPP_IN_CHANNELS,
            out_channels=self.ASPP_OUT_CHANNELS
        )

        self.decoder = _Decoder(
            ll_in_channels=self.ASPP_OUT_CHANNELS,
            ll_reduced_channels=self.DECODER_REDUCED_CHANNELS,
            enc_in_channels=self.DECODER_IN_CHANNELS,
            enc_out_channels=self.DECODER_OUT_CHANNELS
        )

        self.logits = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.DECODER_OUT_CHANNELS, out_channels, 1))
        ]))

    def init_from_tensorflow(self,
                             checkpoint,
                             xception_only=False,
                             init_logits=False):

        import tensorflow as tf

        tfc = _TFConverter(tf, checkpoint)

        self._init_xception(tfc)

        if not xception_only:
            self._init_aspp(tfc)
            self._init_decoder(tfc)

            if init_logits:
                self._init_logits(tfc)

        tfc.warn_if_incomplete(ignore_logits=(not init_logits))

    def forward(self, x):
        ll, x = self.encoder(x)

        x = self.aspp(x)
        x = self.decoder(ll, x)
        x = self.logits(x)

        return x

    def _init_xception(self, tfc):
        # entry
        tfc.set_prefix('xception_65/entry_flow/')

        tfc.conv(self.encoder.conv1, 'conv1_1')
        tfc.conv(self.encoder.conv2, 'conv1_2')
        tfc.xception_block(self.encoder.entry_block1, 'block1/unit_1/xception_module')
        tfc.xception_block(self.encoder.entry_block2, 'block2/unit_1/xception_module')
        tfc.xception_block(self.encoder.entry_block3, 'block3/unit_1/xception_module')

        # main
        tfc.set_prefix('xception_65/middle_flow/')

        for i in range(16):
            pt_block = getattr(self.encoder.main_blocks, 'block{}'.format(i + 1))
            tf_block = 'block1/unit_{}/xception_module'.format(i + 1)
            tfc.xception_block(pt_block, tf_block)

        # exit
        tfc.set_prefix('xception_65/exit_flow/')

        tfc.xception_block(self.encoder.exit_block1, 'block1/unit_1/xception_module')
        tfc.xception_block(self.encoder.exit_block2, 'block2/unit_1/xception_module')

        tfc.set_prefix('')

    def _init_aspp(self, tfc):
        tfc.conv(self.aspp.pool_conv, 'image_pooling')
        tfc.conv(self.aspp.conv_parallel1, 'aspp0')
        tfc.conv(self.aspp.conv_parallel2, 'aspp1', separable=True)
        tfc.conv(self.aspp.conv_parallel3, 'aspp2', separable=True)
        tfc.conv(self.aspp.conv_parallel4, 'aspp3', separable=True)
        tfc.conv(self.aspp.conv, 'concat_projection')

    def _init_decoder(self, tfc):
        tfc.set_prefix('decoder/')

        tfc.conv(self.decoder.ll_conv, 'feature_projection0')
        tfc.conv(self.decoder.conv1, 'decoder_conv0', separable=True)
        tfc.conv(self.decoder.conv2, 'decoder_conv1', separable=True)

    def _init_logits(self, tfc):
        tfc.set_prefix('logits/')

        tfc.conv(self.logits, 'semantic', bias=True)
