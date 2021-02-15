import torch
import torch.nn as nn


def Conv2d1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = False,
) -> torch.nn.Module:
    """
    Implements a point-wise convolution for 2d images, i.e., kernel_size=1x1.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Conv3d1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = False,
) -> torch.nn.Module:
    """
    Implements a point-wise convolution for signals in the group, i.e., kernel_size=1x1x1.
    """
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


class LayerNorm(nn.Module):
    def __init__(
        self,
        num_channels,
        eps=1e-12,
    ):
        """Uses GroupNorm implementation with group=1 for speed reason."""
        super(LayerNorm, self).__init__()
        # we use GroupNorm to implement this efficiently and fast.
        self.layer_norm = torch.nn.GroupNorm(1, num_channels=num_channels, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)
