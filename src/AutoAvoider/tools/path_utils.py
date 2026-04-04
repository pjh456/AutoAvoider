"""Path utilities shared across modules."""

from __future__ import annotations

import glob
import os
from typing import List


def expand_path_arg(path_arg: str) -> List[str]:
    """Expand a path argument to a list of matching paths.

    Supports:
    - wildcard patterns
    - comma-separated lists
    - single path
    """
    if "*" in path_arg:
        return glob.glob(os.path.expanduser(path_arg))
    if "," in path_arg:
        return [os.path.expanduser(p.strip()) for p in path_arg.split(",") if p.strip()]
    return [os.path.expanduser(path_arg)]
