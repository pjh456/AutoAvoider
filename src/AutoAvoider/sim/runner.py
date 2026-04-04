"""Simulation entrypoint (scaffold)."""

from __future__ import annotations

import argparse

from AutoAvoider.common.config import load_yaml
from AutoAvoider.common.logging import setup_logging
from AutoAvoider.sim.mock_env import MockSimEnvironment


def run_sim(config_path: str) -> None:
    logger = setup_logging(name="autoavoider.sim.runner")
    cfg = load_yaml(config_path)
    logger.info("Loaded config: %s", config_path)

    sim_cfg = cfg.get("sensors", {}).get("stereo", {})
    width = int(sim_cfg.get("resolution", {}).get("width", 640))
    height = int(sim_cfg.get("resolution", {}).get("height", 480))

    env = MockSimEnvironment(width=width, height=height)
    obs = env.reset()
    logger.info("Mock env reset, obs keys: %s", list(obs.keys()))

    done = False
    while not done:
        obs, info, done, meta = env.step({"throttle": 0.0, "steer": 0.0})
        if meta.get("step") % 10 == 0:
            logger.info("Step %s info=%s", meta.get("step"), info)

    logger.info("Simulation finished.")


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
