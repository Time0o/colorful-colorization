import torch; torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F

from ..cielab import ABGamut, CIELAB, DEFAULT_CIELAB
from ..util.image import normalize
from .annealed_mean_decode_q import AnnealedMeanDecodeQ
from .deeplab_v3_plus import DeepLabV3Plus
from .get_class_weights import GetClassWeights
from .rebalance_loss import RebalanceLoss
from .soft_encode_ab import SoftEncodeAB
from .vgg_segmentation_network import VGGSegmentationNetwork


class ColorizationNetwork(nn.Module):
    """Wrapper class implementing input encoding, output decoding and class
       rebalancing.

    This class is independent of the concrete underlying network (by default
    the VGG style architecture described by Zhang et al.) so that the latter
    can in principle be exchanged for another network by modifying the
    `base_network` attribute.

    """

    def __init__(self,
                 base_network='vgg',
                 annealed_mean_T=0.38,
                 class_rebal_lambda=None,
                 device='cuda'):
        """
        Construct the network.

        Args:
            base_network (str):
                Underlying base network, currently it is not recommended to
                explictly use this parameter and to only work with the default
                VGG style base network.
            annealed_mean_T (float):
                Annealed mean temperature parameter, should be between 0.0 and
                1.0. Lower values result in less saturated but more spatially
                consistent outputs.
            class_rebal_lambda (float, optional):
                Class rebalancing parameter, class rebalancing is NOT enabled
                by default (i.e. when this is `None`). Zhang et al. recommend
                setting this parameter to 0.5.
            device (str):
                Device on which to run the network (i.e. `'cpu'` or `'cuda'`),
                note that this can not be changed post construction.

        """

        super().__init__()

        self.base_network_id = base_network

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
        """"Network forward pass.

        img (torch.Tensor):
            A tensor of shape `(n, 1, h, w)` where `n` is the size of the
            batch to be predicted and `h` and `w` are image dimensions.
            Must be located on the same device as this network. The
            images should be Lab lightness channels.

        Returns:
            If this network is in training mode: A tuple containing two tensors
            of shape `(n, Q, h, w)`, where `Q` is the number of ab output bins.
            The first element of this tuple is the predicted ab bin distribution
            and the second the soft encoded ground truth ab bin distribution.

            Else, if this model is in evaluation mode: A tensor of shape
            `(n, 1, h, w)` containing the predicted ab channels.

        """

        # label transformation
        if self.training:
            return self._forward_encode(img)
        else:
            return self._forward_decode(img)

    def _forward_encode(self, img):
        l, ab = img[:, :1, :, :], img[:, 1:, :, :]

        l_norm = self._normalize_l(l)

        q_pred = self.base_network(l_norm)

        # downsample and encode labels
        ab = F.interpolate(ab, size=q_pred.shape[2:])
        q_actual = self.encode_ab(ab)

        # rebalancing
        if self.class_rebal_lambda is not None:
            color_weights = self.get_class_weights(q_actual)
            q_pred = self.rebalance_loss(q_pred, color_weights)

        return q_pred, q_actual

    def _forward_decode(self, img):
        l = img

        l_norm = self._normalize_l(l)

        q_pred = self.base_network(l_norm)

        ab_pred = self.decode_q(q_pred)

        return ab_pred

    def _normalize_l(self, l):
        if self.base_network_id == 'vgg':
            l_norm = l - CIELAB.L_MEAN
        elif self.base_network_id == 'deeplab':
            l_norm = normalize(l, (-1, 1))
            l_norm = torch.cat((l_norm,) * 3, dim=1)

        return l_norm

        q_pred = self.base_network(l_norm)
