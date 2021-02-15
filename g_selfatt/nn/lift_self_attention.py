import math

import torch
import torch.nn as nn

import g_selfatt.functional.encoding as encoding_functions
import g_selfatt.nn
from g_selfatt.groups import Group
from g_selfatt.nn.activations import Swish
from g_selfatt.utils import normalize_tensor_one_minone


class LiftSelfAttention(torch.nn.Module):
    def __init__(
        self,
        group: Group,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_heads: int,
        max_pos_embedding: int,
        attention_dropout_rate: float,
    ):
        """
        Creates a lifting self-attention layer with global receptive fields.

        The dimension of the positional encoding is given by mid_channels // 2, where mid_channels // 2 are used to represent
        the row embedding, and mid_channels // 2 channels are used to represent the col embedding.

        Args:
            group: The group to be used, e.g., rotations.
            in_channels:  Number of channels in the input signal
            mid_channels: Number of channels of the hidden representation on which attention is performed.
            out_channels: Number of channels in the output signal
            num_heads: Number of heads in the operation
            max_pos_embedding: The maximum size of the positional embedding to use.
            attention_dropout_rate: Dropout applied to the resulting attention coefficients.
        """
        super().__init__()

        # Define self parameters.
        self.group = group
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        # Define embeddings.
        self.row_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
            g_selfatt.nn.LayerNorm(num_channels=16),
            Swish(),
            torch.nn.Conv2d(in_channels=16, out_channels=self.mid_channels // 2, kernel_size=1),
        )
        self.col_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
            g_selfatt.nn.LayerNorm(num_channels=16),
            Swish(),
            torch.nn.Conv2d(in_channels=16, out_channels=self.mid_channels // 2, kernel_size=1),
        )

        # Create the relative position indices for each element of the group.
        self.initialize_indices(max_pos_embedding)

        # Define linears using convolution 1x1.
        self.query = nn.Conv2d(in_channels, mid_channels * num_heads, kernel_size=1)
        self.key = nn.Conv2d(in_channels, mid_channels * num_heads, kernel_size=1)
        self.value = nn.Conv2d(in_channels, mid_channels * num_heads, kernel_size=1)
        self.wout = nn.Conv3d(mid_channels * num_heads, out_channels, kernel_size=1)

        # Define dropout.
        self.dropout_attention = torch.nn.Dropout(attention_dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        b, c, w, h = x.shape

        # Compute attention scores.
        att_scores = self.compute_attention_scores(x)

        # Normalize to obtain probabilities.
        shape = att_scores.shape
        att_probs = self.dropout_attention(
            torch.nn.Softmax(dim=-1)(att_scores.view(*shape[:-2], -1)).view(shape)
        )

        # Compute value projections.
        v = self.value(x).view(b, self.mid_channels, self.num_heads, w, h)

        # Re-weight values via attention and map to output dimension.
        v = torch.einsum("bhgijkl,bchkl->bchgij", att_probs, v)
        out = self.wout(
            v.contiguous().view(
                b, self.mid_channels * self.num_heads, self.group.num_elements, w, h
            )
        )

        return out

    def compute_attention_scores(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, cin, height, width = x.shape
        sqrt_normalizer = math.sqrt(cin)

        # Compute query and key data.
        q = self.query(x).view(batch_size, self.mid_channels, self.num_heads, height, width)
        k = self.key(x).view(batch_size, self.mid_channels, self.num_heads, height, width)

        # Compute attention scores based on data
        attention_content_scores = torch.einsum("bchij,bchkl->bhijkl", q, k).unsqueeze(2)

        # Compute attention scores based on position
        # B, D // 2, num_attention_heads, W, H
        q_row = q[:, : self.mid_channels // 2, :, :, :]
        q_col = q[:, self.mid_channels // 2 :, :, :, :]

        row_scores = torch.einsum(
            "bchij,gijklc->bhgijkl",
            q_row,
            self.row_embedding(self.row_indices.view(-1, 1, 1, 1)).view(
                self.row_indices.shape + (-1,)
            ),
        )
        col_scores = torch.einsum(
            "bchij,gijklc->bhgijkl",
            q_col,
            self.col_embedding(self.col_indices.view(-1, 1, 1, 1)).view(
                self.col_indices.shape + (-1,)
            ),
        )
        attention_scores = row_scores + col_scores

        # Combine attention scores.
        attention_scores = (
            attention_scores + attention_content_scores.expand_as(attention_scores)
        ) / sqrt_normalizer

        # Return attention scores.
        return attention_scores

    def initialize_indices(
        self,
        max_pos_embedding: int,
    ):
        """
        Creates a set of acted relative positions for each of the elements in the group.
        """

        #  Create 2D relative indices
        indices_1d = normalize_tensor_one_minone(torch.arange(2 * max_pos_embedding - 1))
        indices = torch.stack(
            [
                indices_1d.view(-1, 1).repeat(1, 2 * max_pos_embedding - 1),
                indices_1d.view(1, -1).repeat(2 * max_pos_embedding - 1, 1),
            ],
            dim=0,
        )

        # Get acted versions of the positional encoding indices.
        row_indices = []
        col_indices = []
        for h in self.group.elements:
            Lh_indices = self.group.left_action_on_Rd(h, indices)
            patches = encoding_functions.extract_patches(Lh_indices)
            row_indices_windows = patches[..., 0]
            col_indices_windows = patches[..., 1]
            row_indices.append(row_indices_windows)
            col_indices.append(col_indices_windows)

        self.register_buffer("row_indices", torch.stack(row_indices, dim=0))
        self.register_buffer("col_indices", torch.stack(col_indices, dim=0))
