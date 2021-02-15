import torch


class Swish(torch.nn.Module):
    """
    out = x * sigmoid(x)
    """

    def forward(self, x):
        return x * torch.sigmoid(x)
