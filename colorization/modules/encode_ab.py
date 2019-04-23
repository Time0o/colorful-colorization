import torch


class EncodeAB:
    def __init__(self, cielab):
        self.cielab = cielab

        self.ab_to_q = torch.from_numpy(cielab.ab_to_q).cuda()

    def __call__(self, ab):
        ab_discrete = self._discretize(ab)

        q = self._bin(ab_discrete)

        q = self._expand(q)

        return q.type(ab.dtype)

    def _discretize(self, ab):
        return ((ab - self.cielab.AB_RANGE[0]) / self.cielab.AB_BINSIZE).long()

    def _bin(self, ab):
        n, _, h, w = ab.shape

        # create indices
        ab_ = ab.permute(1, 0, 2, 3).reshape(2, -1)
        a_, b_ = torch.split(ab_, 1)

        # bin ab
        q = self.ab_to_q[a_, b_].reshape(n, 1, h, w)

        return q

    def _expand(self, q):
        n, _, h, w = q.shape

        q_expanded = q.new_zeros(n, self.cielab.gamut.EXPECTED_SIZE, h, w)

        q_expanded.scatter_(1, q, torch.ones_like(q))

        return q_expanded
