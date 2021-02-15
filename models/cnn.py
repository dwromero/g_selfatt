import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        bn_epsilon: float,
        dropout_rate: float,
        use_bias: bool = False,
    ):
        """
        Simple CNN used for rotMNIST. Built upon Cohen & Welling, 2016.
        """
        super().__init__()

        self.sequential = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, num_channels, kernel_size=3, bias=use_bias),
            nn.BatchNorm2d(num_features=num_channels, eps=bn_epsilon),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Layer 2
            nn.Conv2d(num_channels, num_channels, kernel_size=3, bias=use_bias),
            nn.BatchNorm2d(num_features=num_channels, eps=bn_epsilon),
            nn.ReLU(),
            # Max Pooling
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3
            nn.Conv2d(num_channels, num_channels, kernel_size=3, bias=use_bias),
            nn.BatchNorm2d(num_features=num_channels, eps=bn_epsilon),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Layer 4
            nn.Conv2d(num_channels, num_channels, kernel_size=3, bias=use_bias),
            nn.BatchNorm2d(num_features=num_channels, eps=bn_epsilon),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Layer 5
            nn.Conv2d(num_channels, num_channels, kernel_size=3, bias=use_bias),
            nn.BatchNorm2d(num_features=num_channels, eps=bn_epsilon),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Layer 6
            nn.Conv2d(num_channels, num_channels, kernel_size=3, bias=use_bias),
            nn.BatchNorm2d(num_features=num_channels, eps=bn_epsilon),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Layer 7
            nn.Conv2d(num_channels, 10, kernel_size=4, bias=use_bias),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.sequential(x)
        out = out.view(batch_size, 10)
        return out
