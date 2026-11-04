"""CPU checks of the DexBench GA wrist-camera recipe and server against a real 62-joint LeRobot 2.1 root.

    DEXBENCH_TEST_DATASET=/path/to/gr00t_62d_320x180_4cam python -m pytest experiments/dexbench/test_dexbench.py

A few episodes are enough (trim ``meta/episodes.jsonl`` and ``meta/info.json``); no GPU and no model
weights are needed: the server is exercised with a stub model.
"""

import base64
import io
import json
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from PIL import Image

import contract
import serve
from fastwam.datasets.lerobot.robot_video_dataset import canvas_to_model_frames, compose_multi_camera
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[2]
DATASET = os.environ.get("DEXBENCH_TEST_DATASET")
pytestmark = pytest.mark.skipif(not DATASET, reason="set DEXBENCH_TEST_DATASET to a 62-joint DexBench LeRobot root")


def compose_task(work, *overrides):
    register_default_resolvers()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="train", overrides=[
            "task=dexbench_ga_wrist_joint_3cam_384_1e-4",
            f"data.train.dataset_dirs=[{DATASET}]",
            "+data.train.use_text_embed_cache=false",
            f"output_dir={work}",
            *overrides,
        ])


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The composed task config, its training dataset without colour jitter (stats computed on the fly) and
    the run's processor."""
    work = tmp_path_factory.mktemp("run")
    cfg = compose_task(work, "data.train.color_jitter=null")
    misc.register_work_dir(str(work))
    dataset = instantiate(cfg.data.train)
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(
        serve.load_dataset_stats_from_json(str(work / "dataset_stats.json")))
    return cfg, dataset, processor


class StubModel:
    """Records what the server feeds the model and answers zeros (each joint's range midpoint)."""

    device = torch.device("cpu")
    torch_dtype = torch.float32

    def __init__(self):
        self.calls = []

    def encode_prompt(self, prompt):
        self.prompt = prompt
        return torch.zeros(1, 128, 4096), torch.ones(1, 128, dtype=torch.bool)

    def infer_action(self, prompt, input_image, action_horizon, num_video_frames, proprio=None, context=None,
                     context_mask=None, negative_prompt=None, text_cfg_scale=1.0, num_inference_steps=20,
                     sigma_shift=None, seed=None, rand_device="cpu", tiled=False, compile_action_infer=False):
        self.calls.append({"image": input_image, "proprio": proprio, "num_video_frames": num_video_frames,
                           "steps": num_inference_steps, "seed": seed})
        return {"action": torch.zeros(action_horizon, len(contract.JOINT_NAMES))}


def policy(cfg, processor, model):
    return serve.DexBenchFastWAMPolicy(cfg, model, processor, checkpoint="stub", num_inference_steps=10, seed=42,
                                       sigma_shift=None, text_cfg_scale=1.0)


def first_frames(dataset, index=0):
    """The cameras' first frames of sample ``index`` as uint8, exactly as the dataloader decoded them."""
    pixels = dataset.lerobot_dataset[index]["pixel_values"]  # [num_cameras, T, C, H, W] in [0, 1]
    frames = (pixels[:, 0] * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return dict(zip(contract.CAMERAS, frames))


def episode0():
    return pd.read_parquet(Path(DATASET) / "data/chunk-000/episode_000000.parquet")


def test_sample_shapes(run):
    cfg, dataset, _ = run
    assert serve.layout_problems(cfg.data.train) == []
    sample = dataset[0]
    assert tuple(sample["video"].shape) == (3, 9, 384, 320)
    assert sample["video"].min() >= -1 and sample["video"].max() <= 1
    assert tuple(sample["action"].shape) == (32, 62) and tuple(sample["proprio"].shape) == (32, 62)
    assert sample["prompt"].startswith("A video recorded from a robot's point of view")


def test_server_canvas_is_the_training_canvas(run):
    cfg, dataset, processor = run
    served = policy(cfg, processor, StubModel()).canvas(first_frames(dataset))
    assert torch.equal(served[0], dataset[0]["video"][:, 0])


def test_color_jitter_is_one_draw_on_the_training_canvas_only(run, tmp_path):
    cfg, plain, _ = run
    stats = str(Path(misc.get_work_dir()) / "dataset_stats.json")
    jitter = compose_task(tmp_path, f"data.train.pretrained_norm_stats={stats}").data.train
    assert dict(jitter.color_jitter) == {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3}
    jittered = instantiate(jitter)
    assert jittered.color_jitter is not None
    torch.manual_seed(0)
    sample = jittered[0]["video"].permute(1, 0, 2, 3)  # [T, C, H, W] in [-1, 1]
    # The same draw on the composed canvas of the same frames reproduces it: one ColorJitter call per sample.
    pixels = plain.lerobot_dataset[0]["pixel_values"]  # [num_cameras, T, C, H, W] in [0, 1]
    if not plain.lerobot_dataset.presample_images:
        pixels = pixels[:, plain.video_sample_indices]
    canvas = compose_multi_camera(pixels, cfg.data.train.concat_multi_camera)
    torch.manual_seed(0)
    expected = canvas_to_model_frames(jittered.color_jitter(canvas), cfg.data.train.video_size)
    assert torch.equal(sample, expected)
    assert not torch.equal(sample, plain[0]["video"].permute(1, 0, 2, 3))
    # A validation set ignores it.
    evaluation = instantiate(jitter, is_training_set=False)
    assert evaluation.color_jitter is None
    assert torch.equal(evaluation[0]["video"], plain[0]["video"])


def test_state_and_action_units(run):
    cfg, dataset, processor = run
    served = policy(cfg, processor, StubModel())
    frame = episode0()
    state = np.asarray(frame["observation.state"].iloc[0], dtype=np.float32)
    assert torch.allclose(served.normalize_state(state)[0], dataset[0]["proprio"][0], atol=1e-6)
    actions = np.stack(frame["action"].iloc[:32].to_numpy()).astype(np.float32)
    restored = served.denormalize_action(dataset[0]["action"][None])[0]
    assert np.allclose(restored, actions, atol=1e-5)


def test_http_contract(run):
    cfg, dataset, processor = run
    model = StubModel()
    server = serve.make_server(policy(cfg, processor, model), "127.0.0.1", 0, token="secret")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    headers = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
    try:
        info = json.load(urllib.request.urlopen(urllib.request.Request(url + "/info", headers=headers)))
        assert info["joint_names"] == list(contract.JOINT_NAMES) and info["camera_set"] == list(contract.CAMERAS)
        assert info["frame_hw"] == [180, 320] and info["action_horizon"] == 32 and info["num_video_frames"] == 9
        assert info["model_family"] == "fastwam" and info["action_representation"] == {"type": "absolute"}

        def png(frame):
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()

        frames = first_frames(dataset)
        body = {"images": {k: png(v) for k, v in frames.items()}, "prompt": "Take the can",
                "state": np.asarray(episode0()["observation.state"].iloc[0]).tolist()}
        reply = json.load(urllib.request.urlopen(urllib.request.Request(
            url + "/predict", data=json.dumps(body).encode(), headers=headers, method="POST")))
        action = np.asarray(reply["action"])
        assert action.shape == (32, 62) and np.isfinite(action).all()
        call = model.calls[-1]
        assert torch.equal(call["image"][0], dataset[0]["video"][:, 0])
        assert call["num_video_frames"] == 9 and call["steps"] == 10 and call["seed"] == 42
        assert model.prompt.endswith("instruction: Take the can")

        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            urllib.request.urlopen(url + "/info")
        assert unauthorized.value.code == 401
        del body["images"]["camera_wrist_right"]
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(urllib.request.Request(
                url + "/predict", data=json.dumps(body).encode(), headers=headers, method="POST"))
        assert missing.value.code == 400
    finally:
        server.shutdown()
