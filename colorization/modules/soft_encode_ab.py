import math

import torch


class SoftEncodeAB:
    def __init__(self, cielab, neighbours=5, sigma=5.0):
        self.cielab = cielab
        self.q_to_ab = torch.from_numpy(self.cielab.q_to_ab).cuda()

        self.neighbours = neighbours
        self.sigma = sigma

    def __call__(self, ab):
        n, _, h, w = ab.shape

        m = n * h * w

        # find nearest neighbours
        ab_ = ab.permute(1, 0, 2, 3).reshape(2, -1)
        q_to_ab = self.q_to_ab.type(ab_.dtype)

        cdist = torch.cdist(q_to_ab, ab_.t())

        nns = cdist.argsort(dim=0)[:self.neighbours, :]

        # gaussian weighting
        nn_gauss = ab.new_zeros(self.neighbours, m)

        for i in range(self.neighbours):
            nn_gauss[i, :] = self._gauss_eval(
                q_to_ab[nns[i, :], :].t(), ab_, self.sigma)

        nn_gauss /= nn_gauss.sum(dim=0, keepdim=True)

        # expand
        bins = self.cielab.gamut.EXPECTED_SIZE

        q = ab.new_zeros(bins, m)

        q[nns, torch.arange(m).repeat(self.neighbours, 1)] = nn_gauss

        return q.reshape(bins, n, h, w).permute(1, 0, 2, 3)

    @staticmethod
    def _gauss_eval(x, mu, sigma):
        norm = 1 / (2 * math.pi * sigma)

        return norm * torch.exp(-torch.sum((x - mu)**2, dim=0) / (2 * sigma))
