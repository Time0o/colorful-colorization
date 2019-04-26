import torch
import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        return -torch.sum(labels * outputs)
