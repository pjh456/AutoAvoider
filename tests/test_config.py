from __future__ import annotations

from pathlib import Path

from AutoAvoider.common.config import load_yaml


def test_load_yaml_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("a: 1\nb: test\n", encoding="utf-8")

    cfg = load_yaml(str(cfg_path))

    assert cfg.get("a") == 1
    assert cfg.get("b") == "test"
    assert cfg.get("missing", "default") == "default"
