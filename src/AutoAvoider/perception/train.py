"""Perception training entrypoint.

Migration note:
- This is a scaffold to enable incremental porting.
"""

from __future__ import annotations

import argparse

from AutoAvoider.common.config import load_yaml
from AutoAvoider.common.logging import setup_logging


def run_train(config_path: str) -> None:
    logger = setup_logging(name="autoavoider.perception.train")
    cfg = load_yaml(config_path)
    logger.info("Loaded config: %s", config_path)
    logger.info("Model name: %s", cfg.get("model", {}).get("name"))
    logger.info("Training placeholder: migration in progress")


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
