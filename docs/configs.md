# Configs Guide

This document describes the minimal configuration files and their fields. The format is YAML.

## Sim Config (`configs/sim/default.yaml`)
- `scene.name`: Scene identifier (e.g., `empty_world`).
- `scene.seed`: Random seed for deterministic simulation.
- `sensors.stereo.enabled`: Enable stereo camera.
- `sensors.stereo.baseline_m`: Stereo baseline in meters.
- `sensors.stereo.resolution.width`: Image width in pixels.
- `sensors.stereo.resolution.height`: Image height in pixels.
- `output.data_dir`: Output directory for collected data.

## Perception Config (`configs/perception/default.yaml`)
- `model.name`: Model identifier (placeholder for future model registry).
- `model.input.width`: Input image width.
- `model.input.height`: Input image height.
- `model.input.channels`: Input image channels.
- `model.input_keys`: Input tensor keys (e.g., stereo left/right).
- `training.batch_size`: Training batch size.
- `training.epochs`: Training epochs.
- `training.train_split`: Train/val split ratio.
- `training.use_smooth`: Whether to include lateral smoothing output.
- `data.raw_dir`: Raw dataset path.
- `data.processed_dir`: Processed dataset path.
- `data.output_dir`: Output path for trained models.

## Control Config (`configs/control/default.yaml`)
- `control.rate_hz`: Control loop frequency.
- `control.max_speed`: Maximum speed.
- `control.max_steer`: Maximum steering value.
- `safety.emergency_stop_distance_m`: Distance to trigger stop.
- `safety.slow_down_distance_m`: Distance to slow down.

## Vehicle Config (`configs/vehicle/default.yaml`)
- `vehicle.name`: Vehicle name/identifier.
- `vehicle.wheelbase_m`: Wheelbase length in meters.
- `vehicle.track_width_m`: Track width in meters.
- `actuation.interface`: Actuation backend (e.g., `mock`, `gpio`, `can`).
- `actuation.max_throttle`: Maximum throttle.
- `actuation.max_steer`: Maximum steering.
- `sensors.imu`: IMU availability.
- `sensors.odom`: Odometry availability.
