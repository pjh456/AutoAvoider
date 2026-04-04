from __future__ import annotations

import numpy as np

from AutoAvoider.perception.transforms import linear_bin


def test_linear_bin_basic() -> None:
    vec = linear_bin(0.0, bins=15, min_val=-1.0, max_val=1.0)
    assert len(vec) == 15
    assert np.isclose(vec.sum(), 1.0)


def test_linear_bin_clamp() -> None:
    vec = linear_bin(2.0, bins=7, min_val=-1.0, max_val=1.0)
    assert len(vec) == 7
    assert np.isclose(vec.sum(), 1.0)
    assert np.argmax(vec) == 6
