from __future__ import annotations

import pytest


pytest.importorskip("tensorflow")

from AutoAvoider.perception.models.pilot import KerasCategorical


def test_keras_categorical_default_outputs() -> None:
    model = KerasCategorical(resolution=(120, 160), neural_function="default_categorical", use_smooth=False)
    outputs = model.model.outputs
    assert len(outputs) == 2


def test_keras_categorical_optimal_outputs() -> None:
    model = KerasCategorical(resolution=(120, 160), neural_function="optimal_categorical", use_smooth=False)
    outputs = model.model.outputs
    assert len(outputs) == 2
