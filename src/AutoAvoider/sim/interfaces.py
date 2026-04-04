"""Simulation interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class SimEnvironment(ABC):
    """Abstract interface for simulation backends."""

    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment and return initial observations."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """Apply an action and return (obs, info, done, meta)."""
        raise NotImplementedError
