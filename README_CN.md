# JasonDuck 强化学习项目

本项目使用 [mjlab](https://github.com/mujocolab/mjlab)、MuJoCo 和 Warp，
通过 PPO 训练 JasonDuck 的强化学习环境。训练后的策略可在仿真中运行，
也可以导出为 ONNX。

> **Documentation:** [English](README.md)

## 1. 环境要求

- 推荐使用带 NVIDIA GPU、且 NVIDIA 驱动兼容的 Linux 环境进行训练。
  项目没有定义经过验证的 GPU/驱动版本矩阵，也不保证原生 Windows 支持。
- Python **3.12**（`>=3.12,<3.13`）和
  [uv](https://docs.astral.sh/uv/)。
- Git，以及足够的磁盘空间，用于 Python 环境、CUDA 版 PyTorch、
  机器人资源文件、训练日志和模型检查点。

`pyproject.toml` 和 `uv.lock` 固定了 mjlab、Warp、PyTorch 等依赖版本。
Linux `aarch64` 平台通过 uv 选择 PyTorch CUDA 12.9 软件源；其他平台使用配置的
PyPI 软件源。无论使用哪个软件源，都仍然需要可用且兼容的 NVIDIA 驱动。

## 2. 克隆项目并安装

项目地址：[https://github.com/dsydxxy-ctrl/jasonduck_rl](https://github.com/dsydxxy-ctrl/jasonduck_rl)

```bash
git clone https://github.com/dsydxxy-ctrl/jasonduck_rl.git
cd jasonduck_rl
```

如果已经克隆过项目，可在项目根目录更新到最新版本：

```bash
git pull --ff-only
```

如果系统尚未安装 Python 3.12，可使用 uv 安装，然后同步锁定的依赖环境：

```bash
uv python install 3.12
uv sync
```

Linux `aarch64` 设备首次下载 CUDA 版 PyTorch 时，下载文件可能较大。如果 uv
默认 HTTP 超时导致同步中断，可延长超时时间后重试：

```bash
UV_HTTP_TIMEOUT=600 uv sync
```

后续命令都应在仓库根目录执行，并通过 `uv run` 运行，以使用本项目的 Python
环境和命令入口。

## 3. 验证安装

列出 mjlab 已注册的环境：

```bash
uv run list-envs
```

验证项目包可以导入：

```bash
uv run python -c "import mjlab_jasonduck; print('mjlab_jasonduck import OK')"
```

在训练机器上检查 NVIDIA 驱动是否可用：

```bash
nvidia-smi
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda)"
```

使用 GPU 训练时，`torch.cuda.is_available()` 应为 `True`。如果结果为
`False`，请先解决驱动或平台安装问题，再启动长时间训练。

仓库目前没有配置专用的 pytest 测试套件。可以构建 Python 包，作为一项基本检查：

```bash
uv build
```

也可以使用零动作策略启动仿真进行冒烟测试（不需要已训练的检查点）：

```bash
uv run play Mjlab-Velocity-Flat-JasonDuck --agent zero
```

关闭仿真窗口，或按 `Ctrl+C` 停止命令。

## 4. 项目结构

```text
src/mjlab_jasonduck/
  tasks/                  任务注册、环境配置、奖励函数和 MDP
  robot/jasonduck/         JasonDuck MJCF、网格模型及执行器参数
  robot/jasonduck_roller/  滚轮版机器人资源和配置
  robot/microduck/         MicroDuck 资源和配置
  actuator/                自定义执行器集成
  train_cli.py             mjlab train 命令入口封装
  play_hook.py             play 命令的本地检查点选择逻辑
scripts/
  export.py                将训练策略导出为 ONNX
  infer_policy.py          面向 MicroDuck 的 ONNX 推理工具
  hf/                      Hugging Face Jobs 辅助脚本
  train_*.sh               多阶段训练示例
vendor/bam/                 vendored 的 Better Actuator Models 依赖
pyproject.toml              包、依赖和任务入口配置
uv.lock                     锁定的依赖解析结果
```

训练输出默认写入 `logs/rsl_rl/`。日志、检查点和 ONNX 文件都是生成的产物；
需要长期保留的内容请自行备份。

## 5. JasonDuck 可用任务

运行 `uv run list-envs` 可查看当前安装的 mjlab 任务注册表。本仓库注册的
JasonDuck 任务如下：

| 任务 ID | 实验目录 |
|---|---|
| `Mjlab-Velocity-Flat-JasonDuck` | `logs/rsl_rl/jasonduck_velocity/` |
| `Mjlab-Velocity-Rough-JasonDuck` | `logs/rsl_rl/jasonduck_velocity/` |
| `Mjlab-Velocity-Swizzle-JasonDuck` | `logs/rsl_rl/jasonduck_velocity_swizzle/` |
| `Mjlab-StandUp-Flat-JasonDuck` | `logs/rsl_rl/jasonduck_standup/` |
| `Mjlab-StandUp-Rough-JasonDuck` | `logs/rsl_rl/jasonduck_standup/` |
| `Mjlab-RollerStandUp-Flat-JasonDuck` | `logs/rsl_rl/jasonduck_roller_standup/` |
| `Mjlab-GroundPick-Flat-JasonDuck` | `logs/rsl_rl/jasonduck_ground_pick/` |
| `Mjlab-GroundPick-Rough-JasonDuck` | `logs/rsl_rl/jasonduck_ground_pick/` |
| `Mjlab-SitStand-Flat-JasonDuck` | `logs/rsl_rl/jasonduck_sitstand/` |
| `Mjlab-SitStand-Rough-JasonDuck` | `logs/rsl_rl/jasonduck_sitstand/` |
| `Mjlab-BallKick-Flat-JasonDuck` | `logs/rsl_rl/jasonduck_ball_kick_right/` |
| `Mjlab-BallKick-Left-Flat-JasonDuck` | `logs/rsl_rl/jasonduck_ball_kick_left/` |
| `Mjlab-Roulade-Flat-JasonDuck` | `logs/rsl_rl/jasonduck_roulade/` |

每个实验目录下可能有多个运行目录；每次运行会保存独立的 TensorBoard 事件文件
和 `model_*.pt` 检查点。

## 6. 适配机器人或任务

适配新机器人时，建议按以下步骤一起更新相关配置：

1. 在 `src/mjlab_jasonduck/robot/` 下添加 MJCF、网格模型及机器人专属参数。
2. 修改机器人常量和实体配置，指向新模型并映射机身、关节、执行器、初始状态
   和传感器。
3. 更新要使用的任务配置，检查动作关节选择、观测项、奖励、事件和接触检测中
   引用的名称是否存在于新模型。
4. 在 `src/mjlab_jasonduck/tasks/__init__.py` 中注册任务，并确认它出现在
   `uv run list-envs` 输出中。
5. 检查或编译模型，并先进行短时间仿真冒烟测试，再启动长时间训练。

仅修改任务名称并不能完成机器人适配。JasonDuck 的 MJCF、网格模型和 HLS1910
执行器参数与 MicroDuck 的模型及执行器配置不同。

可用于模型检查的工具：

- `src/mjlab_jasonduck/robot/jasonduck/view.py`：在查看器中打开 JasonDuck 模型。
- `src/mjlab_jasonduck/robot/jasonduck/convert_from_urdf.py`：转换 URDF 资源并
  检查生成的 MJCF 是否可以编译。该工具需要网格输入且会生成文件；运行前请查看
  命令行帮助并确认输出路径：

  ```bash
  uv run python src/mjlab_jasonduck/robot/jasonduck/convert_from_urdf.py --help
  ```

修改模型后，先检查几何结构和仿真行为，再进行短时间训练验证配置，确认无误后
再启动长时间训练。

## 7. 训练策略

在仓库根目录启动 PPO 训练：

```bash
uv run train Mjlab-Velocity-Flat-JasonDuck --env.scene.num-envs 4096
```

其他示例：

```bash
uv run train Mjlab-StandUp-Flat-JasonDuck --env.scene.num-envs 4096
uv run train Mjlab-Velocity-Swizzle-JasonDuck --env.scene.num-envs 4096
```

`--env.scene.num-envs` 用于设置并行仿真环境数量。如果显存不足，请调低该值。
训练 CLI 由 mjlab 提供；本项目的 `train` 入口负责封装并转发参数，例如：

```bash
uv run train Mjlab-Velocity-Flat-JasonDuck \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 5
```

上述较小的迭代数适合检查配置，不足以训练出收敛策略。

`scripts/train_jasonduck_getup_then_velocity.sh` 和
`scripts/train_jasonduck_pick_standup_velocity.sh` 提供多阶段训练流程示例。
脚本会依次启动多个任务阶段，但每个阶段都是新的训练运行，不会自动从前一阶段的
检查点继续微调。执行前请先查看脚本参数。

## 8. 观察训练与查找检查点

JasonDuck 任务配置使用 TensorBoard 记录训练数据，默认不会将模型上传到
Weights & Biases。训练运行会创建在相应实验目录下，例如：

```text
logs/rsl_rl/jasonduck_velocity/<run>/
  events.out.tfevents.*       TensorBoard 标量数据
  model_*.pt                  PPO 检查点
```

如果当前环境中提供 TensorBoard，可在项目根目录启动：

```bash
uv run tensorboard --logdir logs/rsl_rl
```

然后打开 TensorBoard 输出的地址（通常为 `http://localhost:6006`）。之后若要
继续查看训练曲线，请保留整个运行目录，包括事件文件和检查点文件。

## 9. 回放已训练策略

如果未指定检查点，项目的 play hook 会在对应任务的实验目录中自动选择最新的
本地 `model_*.pt`：

```bash
uv run play Mjlab-Velocity-Flat-JasonDuck
```

也可以显式指定检查点：

```bash
uv run play Mjlab-Velocity-Flat-JasonDuck \
  --checkpoint-file logs/rsl_rl/jasonduck_velocity/<run>/model_XXXX.pt
```

请使用与检查点训练配置对应的任务。play hook 也支持显式指定 W&B 运行路径来加载
相应运行中的检查点，但本仓库的任务默认不上传模型。`scripts/infer_policy.py`
目前针对 MicroDuck 资源；它不是 JasonDuck 的仿真回放命令。

## 10. 导出 ONNX 策略

在仓库根目录将本地 PPO 检查点导出为 ONNX：

```bash
uv run python scripts/export.py Mjlab-Velocity-Flat-JasonDuck \
  --checkpoint-file logs/rsl_rl/jasonduck_velocity/<run>/model_XXXX.pt \
  --onnx-file logs/rsl_rl/jasonduck_velocity/<run>/policy.onnx
```

将 `<run>` 和 `model_XXXX.pt` 替换为实际运行目录和检查点文件名。指定的检查点
必须存在。若省略 `--onnx-file`，文件默认写入当前目录下的 `output.onnx`。导出器
会将观测归一化合并到 ONNX 计算图，并附加策略元数据。

查看其他导出选项：

```bash
uv run python scripts/export.py --help
```

ONNX 文件是推理产物，不是训练检查点。建议将它与生成它的训练运行保存在一起，
并记录对应的任务和配置。

## 致谢与许可

本项目基于 [Pollen Robotics](https://github.com/pollen-robotics) 的
[microduck_rl](https://github.com/pollen-robotics/microduck_rl)，并使用
[mjlab](https://github.com/mujocolab/mjlab) 和
[BAM（Better Actuator Models）](https://github.com/Rhoban/bam)。Apache License
2.0 的条款和版权声明请见 [LICENSE](LICENSE)。
