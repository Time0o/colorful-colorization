import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        outputs_flat = self._flatten(outputs)
        labels_flat = self._flatten(labels)

        labels_collapsed = labels_flat.max(dim=1)[1]

        return self.loss(outputs_flat, labels_collapsed)

    @staticmethod
    def _flatten(batch):
        n, h, w = batch.shape[:-1]

        batch_reordered = batch.permute(0, 2, 3, 1)

        batch_flat = batch_reordered.contiguous().view(n * h * w, -1)

        return batch_flat
