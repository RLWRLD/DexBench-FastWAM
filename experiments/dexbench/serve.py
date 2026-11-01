"""Serve a DexBench FastWAM checkpoint over the DexBench joint-policy HTTP protocol.

``GET /info`` states the contract: the 62 joint names in the order of ``state`` and ``action``, the
cameras and the frame size per camera the client must send, the action horizon and the sampling
settings. ``POST /predict`` takes ``{"images": {<camera>: <base64 RGB PNG>}, "state": [62 floats],
"prompt": "<task>"}`` and returns ``{"action": [[62 floats] x 32]}``, absolute joint positions in rad.
DexBench-Arena's FastWAM client speaks this protocol.

The server reads everything from the training run directory: ``config.yaml`` (model, data layout and
normalization), ``dataset_stats.json`` and ``checkpoints/weights/step_*.pt``. Each request's frames go
through the training image transforms and the training canvas layout (``to_model_frames``), so the
model sees the canvas it was trained on.

    python experiments/dexbench/serve.py --run-dir runs/<task>/<run_id> --port 5816
"""

import argparse
import base64
import binascii
import inspect
import io
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import contract  # noqa: E402
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT, to_model_frames  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402


def layout_problems(data_cfg) -> list[str]:
    """Why a run's data config is not the DexBench GA wrist-camera layout, or [] when it is."""
    problems = []
    images = data_cfg.shape_meta["images"]
    keys = [item["key"] for item in images]
    if keys != list(contract.CAMERAS):
        problems.append(f"cameras {keys}, expected {list(contract.CAMERAS)}")
    if any(list(item["shape"]) != [3, *contract.FRAME_HW] for item in images):
        problems.append(f"camera frames are not 3x{contract.FRAME_HW[0]}x{contract.FRAME_HW[1]}")
    for group in ("action", "state"):
        if [item["shape"] for item in data_cfg.shape_meta[group]] != [len(contract.JOINT_NAMES)]:
            problems.append(f"{group} is not one 62-D group")
    if data_cfg.concat_multi_camera != "robotwin" or list(data_cfg.video_size) != [384, 320]:
        problems.append("canvas is not the 384x320 robotwin layout")
    return problems


def latest_checkpoint(run_dir: Path) -> Path:
    weights = sorted((run_dir / "checkpoints" / "weights").glob("step_*.pt"))
    if not weights:
        raise FileNotFoundError(f"No checkpoints/weights/step_*.pt under {run_dir}")
    return weights[-1]


def load_model(cfg, checkpoint: Path, device: str, dtype: torch.dtype):
    from hydra.utils import instantiate

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = True
    # The checkpoint holds every MoT weight (loaded strictly below), so the video DiT's pre-trained
    # weights need not be read first. The VAE and text encoder still come from the base model.
    model_cfg.skip_dit_load_from_pretrain = True
    model = instantiate(model_cfg, model_dtype=dtype, device=device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.mot.load_state_dict(payload["mot"], strict=True)
    if model.proprio_encoder is not None:
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    return model.to(device).eval()


class DexBenchFastWAMPolicy:
    """One FastWAM checkpoint with the processor and canvas of the run that trained it."""

    def __init__(self, cfg, model, processor, *, checkpoint: str, num_inference_steps: int, seed: int | None,
                 sigma_shift: float | None, text_cfg_scale: float):
        data = cfg.data.train
        problems = layout_problems(data)
        if problems:
            raise ValueError("Not a DexBench GA wrist-camera run: " + "; ".join(problems))
        self.data = data
        self.model = model
        self.processor = processor
        self.checkpoint = checkpoint
        self.num_inference_steps = int(num_inference_steps)
        self.seed = seed
        self.sigma_shift = sigma_shift
        self.text_cfg_scale = float(text_cfg_scale)
        self.action_horizon = int(data.num_frames) - 1
        self.num_video_frames = self.action_horizon // int(data.action_video_freq_ratio) + 1
        self.model_target = str(cfg.model._target_)
        self._contexts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._requests = 0

    def info(self) -> dict[str, Any]:
        return {
            "model_family": "fastwam",
            "model": self.model_target,
            "checkpoint": self.checkpoint,
            "embodiment": contract.EMBODIMENT,
            "joint_names": list(contract.JOINT_NAMES),
            "camera_set": list(contract.CAMERAS),
            "frame_hw": list(contract.FRAME_HW),
            "canvas": "robotwin 384x320: camera_ego_left 256x320 on top, wrist left/right 128x160 below",
            "fps": contract.FPS,
            "action_horizon": self.action_horizon,
            "num_video_frames": self.num_video_frames,
            "num_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "text_cfg_scale": self.text_cfg_scale,
            "action_representation": {"type": "absolute"},
        }

    def canvas(self, frames: dict[str, np.ndarray]) -> torch.Tensor:
        """``{camera: (H, W, 3) uint8}`` -> the ``[1, 3, 384, 320]`` model input in [-1, 1]."""
        views = []
        for camera in contract.CAMERAS:
            image = torch.from_numpy(np.ascontiguousarray(frames[camera])).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            for transform in self.processor.val_transforms:
                image = transform(image)
            views.append(image)
        video = torch.stack(views, dim=0)  # [num_cameras, 1, C, H, W]
        return to_model_frames(video, self.data.concat_multi_camera, self.data.video_size)  # [1, C, H, W]

    def normalize_state(self, state: np.ndarray) -> torch.Tensor:
        key = self.processor.shape_meta["state"][0]["key"]
        batch = {"state": {key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        return self.processor.normalizer.forward(batch)["state"][key]

    def denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        key = self.processor.shape_meta["action"][0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()

    def _context(self, prompt: str):
        if prompt not in self._contexts:
            self._contexts[prompt] = self.model.encode_prompt(DEFAULT_PROMPT.format(task=prompt))
        return self._contexts[prompt]

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        t0 = time.monotonic()
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        images = request.get("images") or {}
        missing = [camera for camera in contract.CAMERAS if camera not in images]
        if missing:
            raise ValueError(f"images is missing {missing}")
        frames = {camera: decode_png(images[camera], contract.FRAME_HW, camera) for camera in contract.CAMERAS}
        state = np.asarray(request.get("state"), dtype=np.float32)
        if state.shape != (len(contract.JOINT_NAMES),) or not np.isfinite(state).all():
            raise ValueError(f"state must be {len(contract.JOINT_NAMES)} finite joint positions, got {state.shape}")
        with self._lock:
            self._requests += 1
            image = self.canvas(frames).to(device=self.model.device, dtype=self.model.torch_dtype)
            proprio = self.normalize_state(state)
            context, context_mask = self._context(prompt)
            kwargs = dict(prompt=None, input_image=image, action_horizon=self.action_horizon, proprio=proprio,
                          context=context, context_mask=context_mask, text_cfg_scale=self.text_cfg_scale,
                          num_inference_steps=self.num_inference_steps, sigma_shift=self.sigma_shift, seed=self.seed)
            # FastWAM Joint denoises the future video with the actions; the base model takes no frame count.
            if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
                kwargs["num_video_frames"] = self.num_video_frames
            with torch.no_grad():
                result = self.model.infer_action(**kwargs)
            action = self.denormalize_action(result["action"].reshape(1, self.action_horizon, -1))[0]
        if action.shape != (self.action_horizon, len(contract.JOINT_NAMES)) or not np.isfinite(action).all():
            raise RuntimeError(f"model returned an invalid action chunk {action.shape}")
        return {"action": action.tolist(), "request": self._requests, "seconds": round(time.monotonic() - t0, 4)}


def decode_png(b64: str, expected_hw, name: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError, TypeError) as e:
        raise ValueError(f"images.{name}: invalid base64: {e}") from e
    with Image.open(io.BytesIO(raw)) as img:
        if img.mode != "RGB":
            raise ValueError(f"images.{name}: expected an RGB PNG, got mode {img.mode}")
        frame = np.asarray(img, dtype=np.uint8).copy()
    if frame.shape[:2] != tuple(expected_hw):
        raise ValueError(f"images.{name}: frame is {frame.shape[1]}x{frame.shape[0]}, "
                         f"the policy was trained on {expected_hw[1]}x{expected_hw[0]}")
    return frame


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _authorized(self) -> bool:
        token = self.server.token  # type: ignore[attr-defined]
        if token is None or self.headers.get("Authorization", "") == f"Bearer {token}":
            return True
        self._send(401, {"error": "missing or wrong bearer token"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path == "/info":
            self._send(200, self.server.policy.info())  # type: ignore[attr-defined]
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path != "/predict":
            self._send(404, {"error": "not found"})
            return
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            self._send(200, self.server.policy.predict(request))  # type: ignore[attr-defined]
        except Exception as e:  # report to the client instead of dropping the connection
            traceback.print_exc()
            self._send(400 if isinstance(e, ValueError) else 500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def make_server(policy, host: str, port: int, token: str | None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.policy = policy  # type: ignore[attr-defined]
    server.token = token  # type: ignore[attr-defined]
    return server


def main() -> None:
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="training run with config.yaml and dataset_stats.json")
    p.add_argument("--checkpoint", type=Path, help="weights file; default: the run's latest checkpoints/weights/step_*.pt")
    p.add_argument("--host", default="127.0.0.1", help="keep it off 0.0.0.0 unless the port is firewalled")
    p.add_argument("--port", type=int, default=5816)
    p.add_argument("--api-token-file", type=Path, help="file holding a bearer token every request must carry")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42, help="noise seed for every request (default 42)")
    p.add_argument("--sigma-shift", type=float, default=None, help="default: the model's inference shifts")
    p.add_argument("--text-cfg-scale", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = OmegaConf.load(a.run_dir / "config.yaml")
    checkpoint = a.checkpoint or latest_checkpoint(a.run_dir)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[str(cfg.get("mixed_precision", "bf16"))]
    model = load_model(cfg, checkpoint, a.device, dtype)
    processor = OmegaConf.create(OmegaConf.to_container(cfg.data.train.processor, resolve=True))
    from hydra.utils import instantiate

    processor = instantiate(processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(a.run_dir / "dataset_stats.json")))
    policy = DexBenchFastWAMPolicy(cfg, model, processor, checkpoint=str(checkpoint),
                                   num_inference_steps=a.num_inference_steps, seed=a.seed,
                                   sigma_shift=a.sigma_shift, text_cfg_scale=a.text_cfg_scale)
    token = a.api_token_file.read_text().strip() if a.api_token_file else None
    server = make_server(policy, a.host, a.port, token)
    print(f"[dexbench-fastwam] serving http://{a.host}:{a.port} {json.dumps(policy.info())}; "
          f"token {'required' if token else 'off'}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
