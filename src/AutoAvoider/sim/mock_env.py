"""Mock simulation environment."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from AutoAvoider.sim.interfaces import SimEnvironment


def _fake_stereo_frame(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


class MockSimEnvironment(SimEnvironment):
    """A minimal simulation backend for wiring and testing."""

    def __init__(self, width: int = 640, height: int = 480, max_steps: int = 50) -> None:
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self._step = 0

    def reset(self) -> Dict[str, np.ndarray]:
        self._step = 0
        return {
            "stereo/left": _fake_stereo_frame(self.width, self.height),
            "stereo/right": _fake_stereo_frame(self.width, self.height),
        }

    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        self._step += 1
        obs = {
            "stereo/left": _fake_stereo_frame(self.width, self.height),
            "stereo/right": _fake_stereo_frame(self.width, self.height),
        }
        info = {
            "speed": float(action.get("throttle", 0.0)),
            "steer": float(action.get("steer", 0.0)),
        }
        done = self._step >= self.max_steps
        meta = {"step": self._step}
        return obs, info, done, meta
