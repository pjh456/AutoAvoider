"""Simulation entrypoint (scaffold)."""

from __future__ import annotations

import argparse

from AutoAvoider.common.config import load_yaml
from AutoAvoider.common.logging import setup_logging


def run_sim(config_path: str) -> None:
    logger = setup_logging(name="autoavoider.sim.runner")
    cfg = load_yaml(config_path)
    logger.info("Loaded config: %s", config_path)
    logger.info("Simulation scaffold ready. No backend attached yet.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoAvoider simulation runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sim/default.yaml",
        help="Path to sim config YAML",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_sim(args.config)


if __name__ == "__main__":
    main()
