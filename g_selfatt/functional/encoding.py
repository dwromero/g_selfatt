import torch


def extract_patches(indices: torch.Tensor) -> torch.Tensor:
    """Extract patches of relative indices from the full table.

    Args:
        indices: Tensor of shape (2, 2 * height - 1, 2 * width - 1)

    Returns:
        Relative indices tensor of shape (height, width, height, width, 2)
    """
    assert indices.shape[0] == 2
    height = (indices.shape[1] + 1) // 2
    width = (indices.shape[2] + 1) // 2
    dtype = indices.dtype
    indices = indices.float()
    windows = (
        torch.nn.Unfold((height, width))(indices.unsqueeze(0))
        .view((2, height, width, height, width))
        .permute(3, 4, 1, 2, 0)
    )
    windows = windows.flip(0)
    windows = windows.flip(1)
    return windows.type(dtype)
