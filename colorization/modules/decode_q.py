import torch


class DecodeQ:
    def __init__(self, cielab):
        self.cielab = cielab

        self.q_to_ab = torch.from_numpy(cielab.q_to_ab)
        self.q_to_ab_cuda = self.q_to_ab.cuda()

    def __call__(self, q):
        q_discrete = self._collapse(q)

        ab = self._unbin(q_discrete)

        return ab.type(q.dtype)

    def _collapse(self, q):
        return q.max(dim=1, keepdim=True)[1]

    def _unbin(self, q):
        n, _, h, w = q.shape

        q_ = q.permute(1, 0, 2, 3).reshape(-1)

        # dynamically use indices stored on CPU or GPU
        q_to_ab = self.q_to_ab_cuda if q.is_cuda else self.q_to_ab

        # bin ab
        ab = q_to_ab.index_select(0, q_).reshape(n, 2, h, w)

        return ab
