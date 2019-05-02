import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        softmax = log_softmax(outputs, dim=1)

        norm = labels.clone()

        norm[norm != 0] = torch.log(norm[norm != 0])

        return -torch.sum((softmax - norm) * labels) / outputs.shape[0]
