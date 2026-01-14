import torch


def random_range(min: torch.Tensor = -3.14, max: torch.Tensor = 3.14) -> torch.Tensor:
    """
    Generate a random tensor within the specified range.

    Args:
        min (torch.Tensor): Minimum.
        max (torch.Tensor): Maximum.

    Returns:
        torch.Tensor: A random tensor.
    """
    return min + (max - min) * torch.rand(min.shape, device=min.device)