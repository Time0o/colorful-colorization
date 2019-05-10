import torch; torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F

from ..cielab import ABGamut, CIELAB, DEFAULT_CIELAB
from .annealed_mean_decode_q import AnnealedMeanDecodeQ
from .deeplab_v3_plus import DeepLabV3Plus
from .get_class_weights import GetClassWeights
from .rebalance_loss import RebalanceLoss
from .soft_encode_ab import SoftEncodeAB
from .vgg_segmentation_network import VGGSegmentationNetwork


class ColorizationNetwork(nn.Module):
    def __init__(self,
                 base_network='vgg',
                 annealed_mean_T=0.38,
                 class_rebal_lambda=None,
                 device='cuda'):

        super().__init__()

        if base_network == 'vgg':
            self.base_network = VGGSegmentationNetwork(ABGamut.EXPECTED_SIZE)
        elif base_network == 'deeplab':
            self.base_network = DeepLabV3Plus(ABGamut.EXPECTED_SIZE)
        else:
            fmt = "invalid base network type '{}'"
            raise ValueError(fmt.format(base_network))

        self.device = device

        # en-/decoding
        self.encode_ab = SoftEncodeAB(DEFAULT_CIELAB,
                                      device=self.device)

        self.decode_q = AnnealedMeanDecodeQ(DEFAULT_CIELAB,
                                            T=annealed_mean_T,
                                            device=self.device)

        # rebalancing
        self.class_rebal_lambda = class_rebal_lambda

        if class_rebal_lambda is not None:
            self.get_class_weights = GetClassWeights(DEFAULT_CIELAB,
                                                     lambda_=class_rebal_lambda,
                                                     device=self.device)

            self.rebalance_loss = RebalanceLoss.apply

        # move to device
        self.to(self.device)

    def forward(self, img):
        if self.training:
            l, ab = img[:, :1, :, :], img[:, 1:, :, :]
        else:
            l = img

        # normalize lightness
        l_norm = l - CIELAB.L_MEAN

        # prediction
        q_pred = self.base_network(l_norm)

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
