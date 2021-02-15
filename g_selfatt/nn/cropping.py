import torch


class Crop(torch.nn.Module):
    """Crops the images by `crop_size` pixel on each side."""

    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return x[
            ..., self.crop_size : height - self.crop_size, self.crop_size : width - self.crop_size
        ]
