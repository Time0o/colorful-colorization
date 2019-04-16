import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()

        self.interp = F.interpolate
        self.scale_factor = scale_factor

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor)
