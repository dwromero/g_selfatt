import math

import torch
import torch.nn as nn


# **deprecated** Added here only for reference. We implement this with group=z2.
class RdSelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_heads: int,
        max_pos_embedding: int,
        attention_dropout_rate: float,
    ):
        super().__init__()

        # Define self parameters
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.max_pos_embedding = max_pos_embedding

        # Define embeddings
        self.row_embedding = nn.Embedding(2 * max_pos_embedding - 1, mid_channels // 2)
        self.col_embedding = nn.Embedding(2 * max_pos_embedding - 1, mid_channels // 2)
        deltas = torch.arange(max_pos_embedding).view(1, -1) - torch.arange(max_pos_embedding).view(
            -1, 1
        )
        # -- shift the delta to [0, 2 * max_position_embeddings - 1]
        relative_indices = deltas + max_pos_embedding - 1
        self.register_buffer("relative_indices", relative_indices)

        # Define linears
        # If kernel size = 1, then equal to MLP at every position
        self.query = nn.Conv2d(in_channels, mid_channels * num_heads, kernel_size=1)
        self.key = nn.Conv2d(in_channels, mid_channels * num_heads, kernel_size=1)
        self.value = nn.Conv2d(in_channels, mid_channels * num_heads, kernel_size=1)
        self.wout = nn.Conv2d(mid_channels * num_heads, out_channels, kernel_size=1)

        # # Define dropout
        self.dp_att = nn.Dropout(attention_dropout_rate)

    def forward(self, x):
        b, c, w, h = x.shape

        # Compute attention scores.
        att_scores = self.compute_attention_scores(x)
        # Normalize to obtain probabilities.
        shape = att_scores.shape
        att_probs = self.dp_att(nn.Softmax(dim=-1)(att_scores.view(*shape[:-2], -1)).view(shape))

        # Re-weight values via attention and map to output dimension.
        v = self.value(x).view(b, self.mid_channels, self.num_heads, w, h)
        v = torch.einsum("bhijkl,bchkl->bchij", att_probs, v)
        out = self.wout(v.contiguous().view(b, self.mid_channels * self.num_heads, w, h))

        return out

    def compute_attention_scores(self, x):
        bs, cin, height, width = x.shape
        sqrt_normalizer = math.sqrt(cin)

        # compute query and key data
        q = self.query(x).view(bs, self.mid_channels, self.num_heads, height, width)
        k = self.key(x).view(bs, self.mid_channels, self.num_heads, height, width)

        # Compute attention scores based on position
        row_embedding = self.row_embedding(
            self.relative_indices[:width, :width].reshape(-1)
        ).transpose(0, 1)
        col_embedding = self.col_embedding(
            self.relative_indices[:height, :height].reshape(-1)
        ).transpose(0, 1)

        # B, W, H, num_attention_heads, D // 2
        q_row = q[:, : self.mid_channels // 2, :, :, :]
        q_col = q[:, self.mid_channels // 2 :, :, :, :]

        row_scores = torch.einsum("bchij,cik->bhijk", q_row, row_embedding.view(-1, width, width))
        col_scores = torch.einsum("bchij,cjl->bhijl", q_col, col_embedding.view(-1, height, height))

        # -- B, H, W, num_attention_heads, H, W
        attention_scores = row_scores.unsqueeze(-1) + col_scores.unsqueeze(-2)
        attention_scores = attention_scores / sqrt_normalizer

        # Compute attention scores based on data
        attention_content_scores = torch.einsum("bchij,bchkl->bhijkl", q, k)
        attention_content_scores = attention_content_scores / sqrt_normalizer

        # Combine attention scores
        attention_scores = attention_scores + attention_content_scores
        # Return attention scores
        return attention_scores
