"""Perception training entrypoint.

Migration note:
- This is a scaffold to enable incremental porting.
"""

from __future__ import annotations

import argparse

from AutoAvoider.common.config import load_yaml
from AutoAvoider.common.logging import setup_logging
from AutoAvoider.perception.datastore import TubGroup
from AutoAvoider.perception.models.pilot import KerasCategorical
from AutoAvoider.perception.transforms import linear_bin


def _resolve_input_keys(tubgroup: TubGroup, model_cfg: dict) -> list:
    configured = model_cfg.get("input_keys")
    if isinstance(configured, list) and configured:
        return configured
    inputs = set(tubgroup.inputs)
    if "cam/left_image" in inputs and "cam/right_image" in inputs:
        return ["cam/left_image", "cam/right_image"]
    return ["cam/image_array"]


def _validate_input_keys(tubgroup: TubGroup, x_keys: list) -> None:
    inputs = set(tubgroup.inputs)
    missing = [k for k in x_keys if k not in inputs]
    if missing:
        raise KeyError(f"Input keys not found in dataset: {missing}")


def run_train(config_path: str) -> None:
    logger = setup_logging(name="autoavoider.perception.train")
    cfg = load_yaml(config_path)
    logger.info("Loaded config: %s", config_path)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    model_name = model_cfg.get("name", "default_categorical")
    input_cfg = model_cfg.get("input", {})
    resolution = (
        int(input_cfg.get("height", 480)),
        int(input_cfg.get("width", 640)),
    )

    tub_paths = data_cfg.get("raw_dir", "data/raw")
    tubgroup = TubGroup(tub_paths)

    x_keys = _resolve_input_keys(tubgroup, model_cfg)
    _validate_input_keys(tubgroup, x_keys)

    y_keys = ["user/angle", "user/throttle"]
    use_smooth = bool(train_cfg.get("use_smooth", False))
    if use_smooth:
        y_keys.append("user/sth")

    batch_size = int(train_cfg.get("batch_size", 64))
    train_split = float(train_cfg.get("train_split", 0.8))

    def train_record_transform(record: dict) -> dict:
        record["user/angle"] = linear_bin(record["user/angle"])
        if use_smooth:
            record["user/sth"] = linear_bin(record["user/sth"])
        return record

    def val_record_transform(record: dict) -> dict:
        record["user/angle"] = linear_bin(record["user/angle"])
        if use_smooth:
            record["user/sth"] = linear_bin(record["user/sth"])
        return record

    train_gen, val_gen = tubgroup.get_train_val_gen(
        X_keys=x_keys,
        Y_keys=y_keys,
        batch_size=batch_size,
        train_frac=train_split,
        train_record_transform=train_record_transform,
        val_record_transform=val_record_transform,
    )

    model = KerasCategorical(
        resolution=resolution,
        neural_function=model_name,
        use_smooth=use_smooth,
    )

    output_dir = data_cfg.get("output_dir", "data/models")
    model_path = f"{output_dir}/model.h5"

    steps = max(1, len(tubgroup.df) * train_split // batch_size)
    epochs = int(train_cfg.get("epochs", 50))

    logger.info("Training model: %s", model_name)
    model.train(
        train_gen,
        val_gen,
        saved_model_path=model_path,
        epochs=epochs,
        steps=int(steps),
        train_split=train_split,
    )
    logger.info("Training finished, saved to %s", model_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoAvoider perception training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/perception/default.yaml",
        help="Path to perception config YAML",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_train(args.config)


if __name__ == "__main__":
    main()
