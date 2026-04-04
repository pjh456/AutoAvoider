"""Keras training wrappers for perception models."""

from __future__ import annotations

from typing import Any, Tuple

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from AutoAvoider.perception.models.keras import default_categorical, optimal_categorical


class KerasPilot:
    """Training wrapper with load and train utilities."""

    def load(self, model_path: str) -> None:
        self.model = load_model(model_path)

    def train(
        self,
        train_gen: Any,
        val_gen: Any,
        saved_model_path: str,
        epochs: int = 100,
        steps: int = 100,
        train_split: float = 0.8,
        verbose: int = 1,
        min_delta: float = 0.0005,
        patience: int = 5,
        use_early_stop: bool = True,
    ) -> Any:
        save_best = ModelCheckpoint(
            saved_model_path,
            monitor="val_loss",
            verbose=verbose,
            save_best_only=True,
            mode="min",
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode="auto",
        )
        callbacks_list = [save_best]
        if use_early_stop:
            callbacks_list.append(early_stop)

        history = self.model.fit(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=int(steps * (1.0 - train_split) // train_split),
        )
        return history


class KerasCategorical(KerasPilot):
    """Concrete model wrapper selecting the network topology."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (480, 640),
        neural_function: str = "default_categorical",
        use_smooth: bool = False,
    ) -> None:
        if neural_function == "optimal_categorical":
            self.model = optimal_categorical(resolution, use_smooth)
        else:
            self.model = default_categorical(resolution, use_smooth)
