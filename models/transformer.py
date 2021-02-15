import torch
import torch.nn as nn

from g_selfatt.nn import Conv2d1x1, LayerNorm, RdSelfAttention, TransformerBlock


# **Deprecated**. We use group transformer with group z2 instead. Here only for reference.
class Transformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        kernel_size: int,
        image_size: int,
        attention_dropout_rate: float,
        value_dropout_rate: float,
        use_bias: bool,
        norm_type: str,
    ):
        super().__init__()

        Norm = {"BatchNorm": torch.nn.BatchNorm3d, "LayerNorm": LayerNorm}[norm_type]

        def transformer_block_builder(image_size, crop_size):
            attention_layer = RdSelfAttention(
                in_channels=num_channels,
                mid_channels=num_channels // 2,
                out_channels=num_channels,
                num_heads=kernel_size ** 2,
                max_pos_embedding=image_size,
                attention_dropout_rate=attention_dropout_rate,
            )

            return TransformerBlock(
                in_channels=num_channels,
                out_channels=num_channels,
                attention_layer=attention_layer,
                activation_function="ReLU",
                norm_type=norm_type,
                crop_size=crop_size,
                value_dropout_rate=value_dropout_rate,
                dim_mlp_conv=2,
            )

        self.sequential = nn.Sequential(
            Conv2d1x1(in_channels, num_channels),
            Norm(num_channels),
            nn.ReLU(),
            transformer_block_builder(image_size=image_size, crop_size=2),
            transformer_block_builder(image_size=image_size - 4, crop_size=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            transformer_block_builder(image_size=(image_size - 4) // 2, crop_size=2),
            transformer_block_builder(image_size=(image_size - 4) // 2 - 4, crop_size=1),
            transformer_block_builder(image_size=(image_size - 4) // 2 - 6, crop_size=1),
            Conv2d1x1(num_channels, 10),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.sequential(x)
        return out.sum(dim=(-2, -1)).view(batch_size, 10)
