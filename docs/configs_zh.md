# 配置说明

本文档说明当前最小配置文件及字段含义，格式为 YAML。

## 仿真配置（`configs/sim/default.yaml`）
- `scene.name`: 场景标识（例如 `empty_world`）。
- `scene.seed`: 随机种子，用于可复现实验。
- `sensors.stereo.enabled`: 是否启用双目相机。
- `sensors.stereo.baseline_m`: 双目基线（米）。
- `sensors.stereo.resolution.width`: 图像宽度（像素）。
- `sensors.stereo.resolution.height`: 图像高度（像素）。
- `output.data_dir`: 采集数据输出目录。

## 感知配置（`configs/perception/default.yaml`）
- `model.name`: 模型标识（占位，用于后续模型注册）。
- `model.input.width`: 输入图像宽度。
- `model.input.height`: 输入图像高度。
- `model.input.channels`: 输入图像通道数。
- `training.batch_size`: 训练批大小。
- `training.epochs`: 训练轮数。
- `training.train_split`: 训练/验证划分比例。
- `data.raw_dir`: 原始数据路径。
- `data.processed_dir`: 处理后数据路径。
- `data.output_dir`: 训练模型输出路径。

## 控制配置（`configs/control/default.yaml`）
- `control.rate_hz`: 控制循环频率。
- `control.max_speed`: 最大速度。
- `control.max_steer`: 最大转向值。
- `safety.emergency_stop_distance_m`: 紧急制动距离阈值。
- `safety.slow_down_distance_m`: 减速距离阈值。

## 车辆配置（`configs/vehicle/default.yaml`）
- `vehicle.name`: 车辆名称/标识。
- `vehicle.wheelbase_m`: 轴距（米）。
- `vehicle.track_width_m`: 轮距（米）。
- `actuation.interface`: 执行器接口（如 `mock`、`gpio`、`can`）。
- `actuation.max_throttle`: 最大油门。
- `actuation.max_steer`: 最大转向。
- `sensors.imu`: 是否有 IMU。
- `sensors.odom`: 是否有里程计。
