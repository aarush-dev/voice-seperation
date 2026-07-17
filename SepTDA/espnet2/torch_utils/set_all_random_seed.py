"""Seeds Python's, NumPy's, and PyTorch's random number generators for reproducibility."""
import random

import numpy as np
import torch


def set_all_random_seed(seed: int) -> None:
    """Seed the `random`, `numpy`, and `torch` RNGs with the same `seed`.

    Args:
        seed: Random seed to apply to all three RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
