import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        n, _, h, w = outputs.shape

        return -torch.sum(log_softmax(outputs, dim=1) * labels) / (n * h * w)
