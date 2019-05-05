import torch


class AnnealedMeanDecodeQ:
    def __init__(self, cielab, T, device='cuda'):
        self.q_to_ab = torch.from_numpy(cielab.q_to_ab).to(device)

        self.T = T

    def __call__(self, q):
        if self.T == 0:
            # makeing this a special case is somewhat ugly but I have found
            # no way to make this a special case of the branch below (in
            # NumPy that would be trivial)
            ab = self._unbin(self._mode(q))
        else:
            q = self._annealed_softmax(q)

            a = self._annealed_mean(q, 0)
            b = self._annealed_mean(q, 1)
            ab = torch.cat((a, b), dim=1)

        return ab.type(q.dtype)

    def _mode(self, q):
        return q.max(dim=1, keepdim=True)[1]

    def _unbin(self, q):
        _, _, h, w = q.shape

        ab = torch.stack([
            self.q_to_ab.index_select(
                0, q_.flatten()
            ).reshape(h, w, 2).permute(2, 0, 1)

            for q_ in q
        ])

        return ab

    def _annealed_softmax(self, q):
        q = torch.exp(q / self.T)
        q /= q.sum(dim=1, keepdim=True)

        return q

    def _annealed_mean(self, q, d):
        am = torch.tensordot(q, self.q_to_ab[:, d], dims=((1,), (0,)))

        return am.unsqueeze(dim=1)
