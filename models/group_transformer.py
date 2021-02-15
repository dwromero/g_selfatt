import enum
import itertools
from typing import Iterator, Optional, Sequence, Union

import torch
import torch.nn as nn

from g_selfatt.groups import Group
from g_selfatt.nn import (
    Conv3d1x1,
    GroupLocalSelfAttention,
    GroupSelfAttention,
    LayerNorm,
    LiftLocalSelfAttention,
    LiftSelfAttention,
    TransformerBlock,
    activations,
)

# mixed precision
try:
    from torch.cuda.amp import autocast
except ModuleNotFoundError:

    def autocast():
        return lambda f: f


class NormType(str, enum.Enum):
    BATCH_NORM = "BatchNorm"
    LAYERN_NORM = "LayerNorm"


class Activation(str, enum.Enum):
    RELU = "ReLU"
    SWISH = "Swish"


class GroupTransformer(nn.Module):
    def __init__(
        self,
        *,
        group: Group,
        in_channels: int,
        num_channels: int,
        num_heads: int,
        norm_type: str = NormType.LAYERN_NORM,
        block_sizes: Sequence[int],
        expansion_per_block: Optional[Sequence[int]],
        crop_per_layer: Optional[Union[Sequence[int], int]],
        normalize_between_layers: bool,
        maxpool_after_last_block: bool,
        dropout_rate_after_maxpooling: float = 0.0,
        image_size: int,
        patch_size: Optional[int] = None,
        activation_function: Activation,
        attention_dropout_rate: float,
        value_dropout_rate: float,
        input_dropout_rate: float = 0.0,
        num_classes: int = 10,
        whitening_scale: float = 1.41421356,
    ):
        super().__init__()

        use_local_attention = patch_size is not None and patch_size > 0

        num_blocks = len(block_sizes)
        num_layers = sum(block_sizes)
        crop_per_layer = crop_per_layer or 0
        expansion_per_block = expansion_per_block or 1

        if not isinstance(expansion_per_block, int) and len(expansion_per_block) != num_blocks:
            raise ValueError(
                "If 'expansion_per_block' is a list it should be of the same length as the number of blocks."
            )

        if not isinstance(crop_per_layer, int) and len(crop_per_layer) != num_layers:
            raise ValueError(
                "If 'crop_per_layer' is a list it should be of the same length as the number of layers."
            )

        if use_local_attention:
            LiftingSA = LiftLocalSelfAttention
            self.GroupSA = GroupLocalSelfAttention
        else:
            LiftingSA = LiftSelfAttention
            self.GroupSA = GroupSelfAttention

        self.Norm = {
            NormType.BATCH_NORM: torch.nn.BatchNorm3d,
            NormType.LAYERN_NORM: LayerNorm,
        }[NormType(norm_type)]

        self.Activation = {
            Activation.RELU: torch.nn.ReLU,
            Activation.SWISH: activations.Swish,
        }[Activation(activation_function)]

        self.group = group
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.norm_type = norm_type
        self.activation_function = activation_function
        self.attention_dropout_rate = attention_dropout_rate
        self.value_dropout_rate = value_dropout_rate
        self.num_classes = num_classes

        def iterate_values_or_repeat_int(values: Union[int, Sequence[int]]) -> Iterator[int]:
            if isinstance(values, int):
                return itertools.cycle([values])
            else:
                return iter(values)

        crop_iterator = iterate_values_or_repeat_int(crop_per_layer)
        expansion_per_block_iterator = iterate_values_or_repeat_int(expansion_per_block)

        self.input_dropout = nn.Dropout(input_dropout_rate)

        # Lifting layer
        max_pos_embedding = patch_size if use_local_attention else image_size
        self.lifting_self_attention = LiftingSA(
            group,
            in_channels,
            num_channels // 2,
            num_channels,
            num_heads,
            max_pos_embedding,
            attention_dropout_rate,
        )
        self.lifting_normalization = self.Norm(num_channels)
        in_channels = num_channels

        # Group SA layers with a maxpooling between each block
        transformer = []
        for block_idx, block_size in enumerate(block_sizes):
            expansion = next(expansion_per_block_iterator)

            for layer_in_block in range(block_size):
                if layer_in_block == 0:
                    out_channels = expansion * in_channels

                layer_crop_size = next(crop_iterator)
                max_pos_embedding = patch_size if use_local_attention else image_size
                transformer += [
                    self.transformer_layer(
                        max_pos_embedding=max_pos_embedding,
                        crop_size=layer_crop_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                    )
                ]

                if normalize_between_layers:
                    transformer += [
                        self.Norm(out_channels),
                        self.Activation(),
                    ]

                image_size -= 2 * layer_crop_size
                in_channels = out_channels

            is_last_block = block_idx == (len(block_sizes) - 1)
            if (not is_last_block) or maxpool_after_last_block:
                transformer.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
                if dropout_rate_after_maxpooling > 0.0:
                    transformer.append(nn.Dropout(dropout_rate_after_maxpooling))
                image_size //= 2

        transformer += [Conv3d1x1(in_channels, num_classes)]

        self.transformer = nn.Sequential(*transformer)

        self.initialize_network(whitening_scale)

    @autocast()  # required for mixed-precision when training on multiple GPUs.
    def forward(self, x):
        batch_size = x.shape[0]
        out = self.input_dropout(x)
        out = self.lifting_self_attention(out)
        out = self.lifting_normalization(out)
        out = self.Activation()(out)
        out = self.transformer(out)
        return out.sum(dim=(-2, -1)).max(-1).values.view(batch_size, self.num_classes)

    def transformer_layer(self, *, max_pos_embedding, crop_size, in_channels, out_channels):
        attention_layer = self.GroupSA(
            self.group,
            in_channels,
            out_channels // 2,
            out_channels,
            self.num_heads,
            max_pos_embedding,
            self.attention_dropout_rate,
        )

        return TransformerBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            attention_layer=attention_layer,
            norm_type=self.norm_type,
            activation_function=self.activation_function,
            crop_size=crop_size,
            value_dropout_rate=self.value_dropout_rate,
            dim_mlp_conv=3,
        )

    def initialize_network(self, whitening_scale):
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(
                    0,
                    whitening_scale
                    * torch.prod(torch.Tensor(list(m.weight.shape)[1:])) ** (-1 / 2),
                )
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
