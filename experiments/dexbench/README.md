# FastWAM on DexBench RBY1-Wuji2 (GA wrist cameras)

This guide finetunes FastWAM Joint on data collected with the RBY1-M robot and Wuji Hand 2 in DexBench,
using the left ego camera and both wrist cameras, and serves the checkpoint to the DexBench evaluation
client.

This repository's `main` is upstream FastWAM (commit `7faa71108368fbb3b6885649f112af607427a2d4`); this
directory, two configs and two small loader changes add DexBench. Follow the
[environment setup](../../README.md#environment-setup) and
[model preparation](../../README.md#model-preparation) before running the commands below from the
repository root.

| File | Purpose |
| --- | --- |
| [`configs/data/dexbench_ga_wrist_3cam.yaml`](../../configs/data/dexbench_ga_wrist_3cam.yaml) | cameras, 62-D action and state, canvas, normalization |
| [`configs/task/dexbench_ga_wrist_joint_3cam_384_1e-4.yaml`](../../configs/task/dexbench_ga_wrist_joint_3cam_384_1e-4.yaml) | FastWAM Joint, batch, steps, learning rate |
| [`contract.py`](contract.py) | joint order, cameras and frame size; checks a dataset |
| [`train.sh`](train.sh) | upstream ZeRO-2 launcher at global batch 256 |
| [`serve.py`](serve.py) | HTTP server for the DexBench client |
| [`test_dexbench.py`](test_dexbench.py) | CPU checks of the data path and the server |

## Dataset

Use the DexBench 62-joint LeRobot 2.1 sets with four cameras at 320×180 (`lerobot/gr00t_62d_320x180_4cam`
in each GA task's dataset). They are relaid from the simulator's 640×360, 66-joint recordings by
DexBench's `scripts/training/relayout_lerobot_62d.py`, which area-downscales every camera with the same
filter the evaluation client uses. The `action` and `observation.state` columns hold absolute joint
positions in radians, in this order:

| Group | Slice | Joint order |
| --- | --- | --- |
| Left arm | `0:7` | `left_arm_0` through `left_arm_6` |
| Right arm | `7:14` | `right_arm_0` through `right_arm_6` |
| Head | `14:16` | `head_0`, `head_1` |
| Hands | `16:56` | Joint stage, then left/right, then thumb/index/middle/ring/pinky |
| Torso | `56:62` | `torso_0` through `torso_5` |

Slices exclude the end index. The 62 policy joints exclude the four wheels. The four stages are
`cmc_flex`, `cmc_abd`, `mcp`, `ip` for the thumb, and `mcp_flex`, `mcp_abd`, `pip`, `dip` for the other
fingers; each stage lists the five left-hand joints, then the five right-hand joints. The language
instruction is the episode's task from `meta/tasks.jsonl`.

Check a dataset before training:

```bash
python experiments/dexbench/contract.py /path/to/gr00t_62d_320x180_4cam
```

These sets ship `meta/stats.json` but no `meta/episodes_stats.jsonl`; the LeRobot 2.1 loader here reads
them as LeRobot 2.0 does. FastWAM computes its own normalization statistics either way.

## Camera layout

| Camera | LeRobot feature | Canvas slot |
| --- | --- | --- |
| `camera_ego_left` | `observation.images.camera_ego_left` | top, 256×320 |
| `camera_wrist_left` | `observation.images.camera_wrist_left` | bottom left, 128×160 |
| `camera_wrist_right` | `observation.images.camera_wrist_right` | bottom right, 128×160 |

The three cameras form one 384×320 canvas, FastWAM's RoboTwin three-camera layout
(`concat_multi_camera: robotwin`). Each 320×180 frame is resized into its slot with bilinear
antialiasing, without keeping the aspect ratio, as upstream does for RoboTwin. The right ego camera is
not used.

## Finetuning

### 1) Text embedding cache

```bash
python scripts/precompute_text_embeds.py task=dexbench_ga_wrist_joint_3cam_384_1e-4 \
  'data.train.dataset_dirs=[/path/to/gr00t_62d_320x180_4cam]'
```

### 2) Training

```bash
NUM_GPUS=8 bash experiments/dexbench/train.sh \
  'data.train.dataset_dirs=[/path/to/gr00t_62d_320x180_4cam]'
```

List several sets in `data.train.dataset_dirs` to train on several tasks together. The first run
computes the normalization statistics and writes them to `runs/<task>/<run_id>/dataset_stats.json`; set
`data.train.pretrained_norm_stats=<that file>` to reuse them. Additional arguments are Hydra overrides,
for example `max_steps=30000` or `wandb.enabled=true wandb.workspace=<entity>`.

| Setting | Value |
| --- | --- |
| Model | FastWAM Joint (`configs/model/fastwam_joint.yaml`): Wan2.2-TI2V-5B video DiT and the Wan-derived ActionDiT; action queries attend to the future video |
| Window | 33 steps: 32 actions and 9 video frames (every fourth step), the current frame conditioned |
| Action and state | 62-D absolute joint positions, per-joint min/max to [-1, 1] |
| Global batch | 256 = `BATCH_SIZE` 4 per GPU × `NUM_GPUS` 8 × gradient accumulation 8 (`train.sh` derives the accumulation) |
| Updates | 16,000, checkpoints every 2,000 |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.01, gradient clipping 1.0 |
| Learning rate | 1e-4 for the DiTs and the proprio encoder, 5% linear warmup, cosine to 1e-6 |
| Precision | bf16, DeepSpeed ZeRO-2 |
| Flow shifts | video 5.0, action 1.0 (training and inference) |
| Image augmentation | none |

Checkpoints go to `runs/dexbench_ga_wrist_joint_3cam_384_1e-4/<run_id>/checkpoints/weights/step_*.pt`.
`train.sh` launches one node; set `NUM_GPUS` and `BATCH_SIZE` so that their product divides 256.

## Inference server

Run [serve.py](serve.py) on the machine used for model inference, from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/dexbench/serve.py \
  --run-dir runs/dexbench_ga_wrist_joint_3cam_384_1e-4/<run_id> \
  --host 127.0.0.1 --port 5816
```

It loads the run's `config.yaml`, `dataset_stats.json` and latest `checkpoints/weights/step_*.pt`
(`--checkpoint` selects another). `--api-token-file` makes every request carry
`Authorization: Bearer <token>`. Defaults: 10 denoising steps, the model's flow shifts, no classifier-free
guidance, and noise seed 42 for every request.

| Endpoint | Body | Reply |
| --- | --- | --- |
| `GET /info` | | the contract: `joint_names` (the 62 joints above), `camera_set` (the three camera keys in canvas order), `frame_hw` `[180, 320]`, `action_horizon` 32, sampling settings, `action_representation` absolute |
| `POST /predict` | `images` (each camera in `camera_set` as a base64 RGB PNG at `frame_hw`), `state` (62 joint positions in `joint_names` order), `prompt` (the task) | `action`: 32 × 62 absolute joint positions |

The client area-downscales each 640×360 render to 320×180 before encoding; the server builds the
canvas with the training code (`to_model_frames` in `fastwam/datasets/lerobot/robot_video_dataset.py`),
so a frame the client sends reaches the model as the training data did. Keep joint and camera bindings,
task settings, seeds and the execution horizon (how many of the 32 actions to execute before asking
again) in DexBench. The simulator must render the wrist cameras.

## Tests

```bash
DEXBENCH_TEST_DATASET=/path/to/gr00t_62d_320x180_4cam \
  python -m pytest experiments/dexbench/test_dexbench.py
```

A few episodes are enough (trim `meta/episodes.jsonl` and `meta/info.json`). The tests build a training
sample, check that the server's canvas equals the training canvas bit for bit and that the server's
state and action units match training, and run the HTTP endpoints against a stub model. They need
neither a GPU nor the model weights.
