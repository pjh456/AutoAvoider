# AutoAvoider

![Status](https://img.shields.io/badge/status-refactoring-blue)
![Focus](https://img.shields.io/badge/focus-simulation%20first-6c5ce7)
![Vision](https://img.shields.io/badge/vision-stereo-00b894)

AutoAvoider is a simulation-first stereo-vision obstacle avoidance project focused on scalable data collection, model training, and safe deployment to a small vehicle platform. The goal is to build a repeatable pipeline that starts in a virtual environment and transfers to real-world driving with strong generalization.

**中文文档**: [docs/README_zh.md](docs/README_zh.md)

## Core Goals
1. Build a physics-based virtual environment for stereo data generation.
2. Collect high-quality stereo + control data for obstacle avoidance.
3. Train and evaluate models that generalize across scenes and lighting.
4. Provide a clean path from simulation to real vehicle control.

## Repository Structure (Planned)
- `docs/` Project docs, design notes, and usage guides.
- `configs/` Training, simulation, and deployment configs.
- `scripts/` One-click scripts for data collection and training.
- `data/` Dataset storage (excluded from git).
- `sim/` Simulation environment, scene setup, sensors.
- `perception/` Models, training code, evaluation.
- `control/` Planning, decision, and control logic.
- `vehicle/` Real-vehicle adapters and drivers.
- `services/` APIs or control services if needed.
- `web/` Optional UI or dashboard.
- `tools/` Utilities for visualization and data processing.
- `tests/` Unit and integration tests.

## Quick Start (Placeholder)
This project is under active refactor. Setup steps will be added once the initial scaffolding is complete.

## Data Policy
Large datasets, trained weights, and logs should live outside git. Use `data/` and `models/` locally or external storage.

## Roadmap
1. Establish simulation backbone and stereo camera model.
2. Define data schema and collection pipeline.
3. Implement baseline avoidance model.
4. Evaluate sim-to-real transfer and iterate.

## License
TBD
