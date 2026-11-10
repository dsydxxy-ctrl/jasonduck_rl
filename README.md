# JasonDuck RL

Reinforcement-learning environments for JasonDuck, trained with PPO using
[mjlab](https://github.com/mujocolab/mjlab), MuJoCo, and Warp. Policies can be
played in simulation and exported to ONNX.

> **Documentation:** [简体中文](README_CN.md)

## 1. Prerequisites

- Linux with an NVIDIA GPU and a compatible NVIDIA driver is the recommended
  setup for training. The repository does not define a supported GPU/driver
  matrix or guarantee native Windows support.
- Python **3.12** (`>=3.12,<3.13`) and [uv](https://docs.astral.sh/uv/).
- Git and enough disk space for the Python environment, CUDA-enabled PyTorch
  wheels, robot assets, training logs, and checkpoints.

The project pins mjlab, Warp, and PyTorch versions in `pyproject.toml` and
`uv.lock`. On Linux `aarch64`, uv selects PyTorch's CUDA 12.9 package index;
other platforms use the configured PyPI source. This does not replace the need
for a working NVIDIA driver.

## 2. Clone and install

Repository: [https://github.com/dsydxxy-ctrl/jasonduck_rl](https://github.com/dsydxxy-ctrl/jasonduck_rl)

```bash
git clone https://github.com/dsydxxy-ctrl/jasonduck_rl.git
cd jasonduck_rl
```

For an existing checkout, update it from the project root:

```bash
git pull --ff-only
```

Install Python 3.12 if it is not already available, then synchronize the
locked environment:

```bash
uv python install 3.12
uv sync
```

On Linux `aarch64` devices, the initial CUDA-enabled PyTorch download can be
large. If uv's default HTTP timeout interrupts the download, retry with a
longer timeout:

```bash
UV_HTTP_TIMEOUT=600 uv sync
```

Run project commands from the repository root using `uv run`; this selects the
project environment and its installed entry points.

## 3. Verify the installation

List the environments registered with mjlab:

```bash
uv run list-envs
```

Confirm the package imports:

```bash
uv run python -c "import mjlab_jasonduck; print('mjlab_jasonduck import OK')"
```

On a training machine, check that the NVIDIA driver is visible:

```bash
nvidia-smi
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda)"
```

`torch.cuda.is_available()` should be `True` for GPU training. If it is false,
resolve the driver/platform installation before starting a long run.

The repository does not currently define a dedicated pytest suite. To check
that the package can be built:

```bash
uv build
```

For a simulation smoke test, launch a task with a zero-action policy (no
trained checkpoint is required):

```bash
uv run play Mjlab-Velocity-Flat-JasonDuck --agent zero
```

Close the simulation window or stop the command with `Ctrl+C`.

## 4. Project layout

```text
src/mjlab_jasonduck/
  tasks/                  Task registration, environment configs, rewards, MDP
  robot/jasonduck/         JasonDuck MJCF, meshes, and actuator parameters
  robot/jasonduck_roller/  Roller-equipped robot assets/configuration
  robot/microduck/         MicroDuck assets/configuration
  actuator/                Custom actuator integration
  train_cli.py             mjlab train entry-point wrapper
  play_hook.py             Local checkpoint selection for play
scripts/
  export.py                Export a trained policy to ONNX
  infer_policy.py          MicroDuck-specific ONNX inference utility
  hf/                      Hugging Face Jobs helpers
  train_*.sh               Multi-stage training examples
vendor/bam/                 Vendored Better Actuator Models dependency
pyproject.toml              Package, dependencies, and task entry point
uv.lock                     Locked dependency resolution
```

Training outputs are written under `logs/rsl_rl/` by default. Logs,
checkpoints, and ONNX exports are generated artifacts; keep backups of anything
you need to retain.

## 5. Available JasonDuck tasks

Run `uv run list-envs` for the registry provided by the installed mjlab
environment. JasonDuck task IDs registered by this repository include:

| Task ID | Experiment directory |
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

Experiment directories may contain multiple runs. Each run stores its own
TensorBoard event files and `model_*.pt` checkpoints.

## 6. Adapt the robot or task

For a new robot, adapt the model in coordinated steps:

1. Add the MJCF, meshes, and model-specific parameters under
   `src/mjlab_jasonduck/robot/`.
2. Update the robot constants/entity configuration to point to the model and
   map its bodies, joints, actuators, initial state, and sensors.
3. Update the selected task configuration and check that action joint
   selection, observation terms, rewards, events, and contact references use
   names that exist in the new model.
4. Register the task in `src/mjlab_jasonduck/tasks/__init__.py` and verify that
   it appears in `uv run list-envs`.
5. Inspect/compile the model and run a short simulation smoke test before
   beginning a longer training run.

Changing only a task name does not adapt a robot model. For JasonDuck, the
MJCF/meshes and HLS1910 actuator parameters are distinct from the MicroDuck
assets and actuators.

Useful model inspection tools:

- `src/mjlab_jasonduck/robot/jasonduck/view.py` opens the JasonDuck model in a
  viewer.
- `src/mjlab_jasonduck/robot/jasonduck/convert_from_urdf.py` converts URDF
  assets and performs an MJCF compile check. It requires mesh input and writes
  generated files. Review its command-line help and output paths before use:

  ```bash
  uv run python src/mjlab_jasonduck/robot/jasonduck/convert_from_urdf.py --help
  ```

After adapting a model, first check its geometry and simulation behavior, then
run a short training experiment before committing to a long run.

## 7. Train a policy

Start a PPO run from the repository root:

```bash
uv run train Mjlab-Velocity-Flat-JasonDuck --env.scene.num-envs 4096
```

Other examples:

```bash
uv run train Mjlab-StandUp-Flat-JasonDuck --env.scene.num-envs 4096
uv run train Mjlab-Velocity-Swizzle-JasonDuck --env.scene.num-envs 4096
```

`--env.scene.num-envs` controls the number of parallel simulation
environments. Reduce it if the available GPU memory is insufficient. mjlab
owns the training CLI; this project wraps its `train` entry point and forwards
training arguments, for example:

```bash
uv run train Mjlab-Velocity-Flat-JasonDuck \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 5
```

The short iteration limit above is useful for checking a configuration, not
for producing a converged policy.

The scripts `scripts/train_jasonduck_getup_then_velocity.sh` and
`scripts/train_jasonduck_pick_standup_velocity.sh` provide multi-stage
training recipes. They run multiple task stages, but each stage starts a new
training run; they do not automatically fine-tune the next stage from the
previous stage's checkpoint. Review the script arguments before running them.

## 8. Observe training and locate checkpoints

JasonDuck task configurations use TensorBoard logging. They do not upload
models to Weights & Biases by default. Training runs are created below the
experiment directory, for example:

```text
logs/rsl_rl/jasonduck_velocity/<run>/
  events.out.tfevents.*       TensorBoard scalars
  model_*.pt                  PPO checkpoints
```

If TensorBoard is available in the environment, start it from the project root:

```bash
uv run tensorboard --logdir logs/rsl_rl
```

Then open the URL printed by TensorBoard (typically
`http://localhost:6006`). To resume analysis later, keep the full run directory,
including its event files and checkpoint files.

## 9. Play a trained policy

With no checkpoint option, the project play hook selects the newest local
`model_*.pt` in the task's experiment directory:

```bash
uv run play Mjlab-Velocity-Flat-JasonDuck
```

Select a particular checkpoint explicitly:

```bash
uv run play Mjlab-Velocity-Flat-JasonDuck \
  --checkpoint-file logs/rsl_rl/jasonduck_velocity/<run>/model_XXXX.pt
```

Use the task matching the checkpoint's training configuration. The play hook
also accepts an explicit W&B run path when using a checkpoint from such a run,
but repository task configs do not upload models by default. The script
`scripts/infer_policy.py` currently targets MicroDuck assets; it is not the
JasonDuck simulation playback command.

## 10. Export a policy to ONNX

Export a local PPO checkpoint from the repository root:

```bash
uv run python scripts/export.py Mjlab-Velocity-Flat-JasonDuck \
  --checkpoint-file logs/rsl_rl/jasonduck_velocity/<run>/model_XXXX.pt \
  --onnx-file logs/rsl_rl/jasonduck_velocity/<run>/policy.onnx
```

Replace `<run>` and `model_XXXX.pt` with the actual run directory and checkpoint
filename. The checkpoint must exist. `--onnx-file` defaults to `output.onnx`
in the current directory if omitted. The exporter includes observation
normalization in the ONNX graph and attaches policy metadata.

Inspect available export options with:

```bash
uv run python scripts/export.py --help
```

The exported ONNX policy is an inference artifact, not a training checkpoint.
Keep it with the run that produced it and record the task/configuration used.

## Acknowledgments and license

This project builds on [microduck_rl](https://github.com/pollen-robotics/microduck_rl)
by [Pollen Robotics](https://github.com/pollen-robotics), [mjlab](https://github.com/mujocolab/mjlab),
and [BAM (Better Actuator Models)](https://github.com/Rhoban/bam). See
[LICENSE](LICENSE) for the Apache License 2.0 terms and copyright notices.
