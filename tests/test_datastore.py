from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from AutoAvoider.perception.datastore import Tub


def test_tub_minimal_record(tmp_path: Path) -> None:
    tub_dir = tmp_path / "tub"
    tub_dir.mkdir()
    meta = {
        "inputs": ["cam/image_array", "user/angle"],
        "types": ["image_array", "float"],
    }
    (tub_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    record = {
        "cam/image_array": "image_0.jpg",
        "user/angle": 0.0,
    }
    (tub_dir / "record_0.json").write_text(json.dumps(record), encoding="utf-8")

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv is required for this test") from exc
    cv2.imwrite(str(tub_dir / "image_0.jpg"), img)

    tub = Tub(str(tub_dir))
    data = tub.get_record(0)

    assert "cam/image_array" in data
    assert data["cam/image_array"].shape[0] == 10
    assert data["user/angle"] == 0.0
