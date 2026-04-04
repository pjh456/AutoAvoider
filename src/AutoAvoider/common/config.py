"""Shared config loading utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Config:
    """Lightweight config container."""

    data: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def load_yaml(path: str) -> Config:
    """Load a YAML config file and return a Config wrapper."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return Config(data=data)
