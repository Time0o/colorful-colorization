import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        labels_collapsed = labels.max(dim=1)[1]

        return self.loss(outputs, labels_collapsed)
