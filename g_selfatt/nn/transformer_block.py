import torch
import torch.nn as nn

from g_selfatt.nn import activations
from g_selfatt.nn.cropping import Crop
from g_selfatt.nn.layers import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_layer: nn.Module,
        norm_type: str,
        activation_function: str,
        crop_size: int,
        value_dropout_rate: float,
        dim_mlp_conv: int,
    ):
        """
        Creates a transformer block as:

        input
        |---------------------------|
        norm                        |
        activation                  |
        self-attention layer        |
        norm                        |
        activation                  |
        dropout (value_rate)        |
        |                           |
        + <-------------------------|
        |---------------------------|
        point-wise linear           |
        norm                        |
        point-wise linear           |
        norm                        |
        activation
        + <-------------------------|
        |
        output

        Args:
            in_channels: Number of channels in the input signal
            out_channels: Number of output (and hidden) channels of the block
            attention_layer: The type of attention layer to be used, e.g., lifting or group self-attention.
            norm_type: The normalization type to use, e.g., LayerNorm.
            activation_function: The activation function of the block, e.g., ReLU
            crop_size: How much must be cropped at each side of the output.
            value_dropout_rate: Dropout on the resulting representation of the self-attention layer (See graphical description above).
            dim_mlp_conv: The dimensionality of the MLP to use, e.g., 2 for spatial signal, 3 for group signal.
        """
        super().__init__()
        self.crop_size = crop_size
        self.crop = Crop(crop_size)

        Norm = {
            "BatchNorm": torch.nn.BatchNorm3d,
            "LayerNorm": LayerNorm,
        }[norm_type]

        Conv = {
            2: nn.Conv2d,
            3: nn.Conv3d,
        }[dim_mlp_conv]

        ActivationFunction = {
            "ReLU": torch.nn.ReLU,
            "Swish": activations.Swish,
        }[activation_function]

        self.attention = nn.Sequential(
            Norm(in_channels),
            ActivationFunction(),
            attention_layer,
            Norm(out_channels),
            ActivationFunction(),
            nn.Dropout(value_dropout_rate),
        )

        self.mlp = nn.Sequential(
            Conv(out_channels, out_channels // 2, kernel_size=1),
            Norm(out_channels // 2),
            ActivationFunction(),
            Conv(out_channels // 2, out_channels, kernel_size=1),
            Norm(out_channels),
        )

        shortcut = []
        if in_channels != out_channels:
            shortcut.append(Conv(in_channels, out_channels, kernel_size=1))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        # Crop then sum consumes less memory.
        out = self.crop(self.attention(x)) + self.crop(self.shortcut(x))
        out = self.mlp(out) + out
        return out
