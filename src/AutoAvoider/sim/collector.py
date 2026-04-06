"""Simulation data collector."""

from __future__ import annotations

import json
import os
import time
from typing import Dict

import cv2

from AutoAvoider.common.logging import setup_logging
from AutoAvoider.sim.interfaces import SimEnvironment

logger = setup_logging(name="autoavoider.sim.collector")


class TubWriter:
    """Write stereo data and control labels in tub format."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        self._ix = 0

    def write_meta(self) -> None:
        meta = {
            "inputs": [
                "cam/left_image",
                "cam/right_image",
                "user/angle",
                "user/throttle",
            ],
            "types": [
                "image_array",
                "image_array",
                "float",
                "float",
            ],
        }
        meta_path = os.path.join(self.root_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def write_record(self, obs: Dict, action: Dict[str, float]) -> None:
        left = obs["stereo/left"]
        right = obs["stereo/right"]

        left_name = f"image_{self._ix}_left.jpg"
        right_name = f"image_{self._ix}_right.jpg"

        cv2.imwrite(os.path.join(self.root_dir, left_name), left)
        cv2.imwrite(os.path.join(self.root_dir, right_name), right)

        record = {
            "cam/left_image": left_name,
            "cam/right_image": right_name,
            "user/angle": float(action.get("steer", 0.0)),
            "user/throttle": float(action.get("throttle", 0.0)),
        }
        record_path = os.path.join(self.root_dir, f"record_{self._ix}.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f)

        self._ix += 1


def collect(env: SimEnvironment, output_dir: str, max_records: int) -> str:
    """Collect data from env and write to a new tub directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    tub_dir = os.path.join(output_dir, f"tub_{timestamp}")
    writer = TubWriter(tub_dir)
    writer.write_meta()

    obs = env.reset()
    for _ in range(max_records):
        action = {"throttle": 0.0, "steer": 0.0}
        writer.write_record(obs, action)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    logger.info("Collected %s records into %s", max_records, tub_dir)
    return tub_dir
