import torch


class AnnealedMeanDecodeQ:
    def __init__(self, cielab, T):
        self.cielab = cielab
        self.T = T

        self.q_to_ab = torch.from_numpy(cielab.q_to_ab)
        self.q_to_ab_cuda = self.q_to_ab.cuda()

    def __call__(self, q):
        q_discrete = self._collapse(q)

        ab = self._unbin(q_discrete)

        return ab.type(q.dtype)

    def _collapse(self, q):
        if self.T == 0:
            return q.max(dim=1, keepdim=True)[1]
        else:
            q = torch.exp(torch.log(q) / self.T)
            q /= q.sum(dim=1, keepdim=True)

            return q.mean(dim=1, keepdim=True)

    def _unbin(self, q):
        _, _, h, w = q.shape

        # dynamically use indices stored on CPU or GPU
        q_to_ab = self.q_to_ab_cuda if q.is_cuda else self.q_to_ab

        # bin ab
        ab = torch.stack([
            q_to_ab.index_select(0, q_.flatten()).reshape(h, w, 2).permute(2, 0, 1)
            for q_ in q
        ])

        return ab
