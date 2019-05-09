import torch


class GetClassWeights:
    def __init__(self, cielab, lambda_=0.5, device='cuda'):
        prior = torch.from_numpy(cielab.gamut.prior).to(device)

        uniform = torch.zeros_like(prior)
        uniform[prior > 0] = 1 / (prior > 0).sum().type_as(uniform)

        self.weights = 1 / ((1 - lambda_) * prior + lambda_ * uniform)
        self.weights /= torch.sum(prior * self.weights)

    def __call__(self, ab_actual):
        return self.weights[ab_actual.argmax(dim=1, keepdim=True)]
