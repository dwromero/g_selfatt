import torch


def max_pooling_rd(
    input: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int = 1,
) -> torch.Tensor:
    """
    Performs max-pooling on a tensor in Rd.
    """
    out = torch.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding)
    return out


def average_pooling_rd(
    input: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int = 1,
) -> torch.Tensor:
    """
    Performs average-pooling on a tensor in Rd.
    """
    out = torch.nn.functional.avg_pool2d(
        input, kernel_size=kernel_size, stride=stride, padding=padding
    )
    return out


def max_pooling_rd_on_G(
    input: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int = 1,
) -> torch.Tensor:
    """
    Performs max-pooling on the spatial dimensions of a tensor defined on the group.
    """
    input_size = input.size()
    out = input.contiguous().view(
        input_size[0], input_size[1] * input_size[2], input_size[3], input_size[4]
    )
    out = torch.max_pool2d(out, kernel_size=kernel_size, stride=stride, padding=padding)
    out = out.contiguous().view(
        input_size[0], input_size[1], input_size[2], out.size()[2], out.size()[3]
    )
    return out


def average_pooling_rd_on_G(
    input: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int = 1,
) -> torch.Tensor:
    """
    Performs average-pooling on the spatial dimensions of a tensor defined on the group.
    """
    input_size = input.size()
    out = input.contiguous().view(
        input_size[0], input_size[1] * input_size[2], input_size[3], input_size[4]
    )
    out = torch.nn.functional.avg_pool2d(
        out, kernel_size=kernel_size, stride=stride, padding=padding
    )
    out = out.contiguous().view(
        input_size[0], input_size[1], input_size[2], out.size()[2], out.size()[3]
    )
    return out
