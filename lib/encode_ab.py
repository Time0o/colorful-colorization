import torch


class EncodeAB:
    def __init__(self, cielab):
        self.cielab = cielab

        self.ab_to_q = torch.from_numpy(cielab.ab_to_q)
        self.ab_to_q_cuda = self.ab_to_q.cuda()

    def __call__(self, ab):
        ab = self._discretize(ab)

        q = self._bin(ab)

        q = self._expand(q)

        return q

    def _discretize(self, ab):
        return ((ab - self.cielab.AB_RANGE[0]) / self.cielab.AB_BINSIZE).long()

    def _bin(self, ab):
        n, _, h, w = ab.shape

        # create indices
        ab_ = ab.permute(1, 0, 2, 3).reshape(2, -1)
        a_, b_ = torch.split(ab_, 1)

        # dynamically use indices stored on CPU or GPU
        ab_to_q = self.ab_to_q_cuda if ab.is_cuda else self.ab_to_q

        # bin ab
        q = ab_to_q[a_, b_].reshape(n, 1, h, w)

        return q

    def _expand(self, q):
        n, _, h, w = q.shape

        q_expanded = q.new_empty(n, self.cielab.gamut.EXPECTED_SIZE, h, w)

        q_expanded.scatter_(1, q, torch.ones_like(q))

        return q_expanded
