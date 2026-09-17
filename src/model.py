import torch.nn as nn


def build_net(n_input):
    return nn.Sequential(
        nn.Linear(n_input, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 150),
    )


class BaselineModel(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.net = build_net(n_input)

    def forward(self, x, _mask):
        return self.net(x)


class MaskedModel(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.net = build_net(n_input)

    def forward(self, x, mask):
        logits = self.net(x)
        return logits.masked_fill(mask == 0, -1e9)