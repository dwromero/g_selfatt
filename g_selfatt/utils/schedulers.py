import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def linear_warmup_cosine_lr_scheduler(
    optimizer,
    warmup_time_ratio: float,
    T_max: int,
) -> torch.optim.lr_scheduler:
    """
    Creates a cosine learning rate scheduler with a linear warmup time determined by warmup_time_ratio.
    The warm_up increases linearly the learning rate from zero up to the defined learning rate.

    Args:
        warmup_time_ratio: Ratio in normalized percentage, e.g., 10% = 0.1, of the total number of iterations (T_max)
        T_max: Number of iterations
    """
    T_warmup = int(T_max * warmup_time_ratio)

    def lr_lambda(epoch):
        # linear warm up
        if epoch < T_warmup:
            return epoch / T_warmup
        else:
            progress_0_1 = (epoch - T_warmup) / (T_max - T_warmup)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress_0_1))
            return cosine_decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
