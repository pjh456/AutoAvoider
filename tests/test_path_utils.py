from __future__ import annotations

from AutoAvoider.tools.path_utils import expand_path_arg


def test_expand_path_arg_single() -> None:
    paths = expand_path_arg("data/raw")
    assert paths == ["data/raw"]


def test_expand_path_arg_comma() -> None:
    paths = expand_path_arg("a,b")
    assert paths == ["a", "b"]
