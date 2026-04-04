"""Training data transforms."""

from __future__ import annotations

import numpy as np


def linear_bin(value: float, bins: int = 15, min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
    """Convert a continuous value to a one-hot bin vector."""
    value = max(min_val, min(max_val, value))
    span = max_val - min_val
    if span <= 0:
        raise ValueError("Invalid range for binning")
    bin_index = int(round((value - min_val) / span * (bins - 1)))
    one_hot = np.zeros(bins, dtype=np.float32)
    one_hot[bin_index] = 1.0
    return one_hot
