from __future__ import annotations

import json
from pathlib import Path

from AutoAvoider.sim.collector import collect
from AutoAvoider.sim.mock_env import MockSimEnvironment


def test_collect_creates_tub(tmp_path: Path) -> None:
    env = MockSimEnvironment(width=16, height=12, max_steps=2)
    tub_dir = collect(env, output_dir=str(tmp_path), max_records=3)

    tub_path = Path(tub_dir)
    assert (tub_path / "meta.json").exists()
    assert (tub_path / "record_0.json").exists()

    meta = json.loads((tub_path / "meta.json").read_text(encoding="utf-8"))
    assert "inputs" in meta

    record = json.loads((tub_path / "record_0.json").read_text(encoding="utf-8"))
    assert "cam/left_image" in record
    assert "cam/right_image" in record
    assert (tub_path / record["cam/left_image"]).exists()
    assert (tub_path / record["cam/right_image"]).exists()
