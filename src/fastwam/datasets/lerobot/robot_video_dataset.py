import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
import torchvision.transforms.v2 as transforms_v2
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def compose_multi_camera(video: torch.Tensor, concat_multi_camera: Optional[str]) -> torch.Tensor:
    """Lay out ``[num_cameras, T, C, H, W]`` frames as one ``[T, C, H', W']`` video.

    ``"robotwin"`` places the first camera in a 256x320 top slot and the other two in 128x160 slots
    below it (a 384x320 canvas); ``"horizontal"``/``"vertical"`` concatenate the cameras side by side.
    Inference servers call this to build exactly the canvas the model was trained on.
    """
    num_cameras = video.shape[0]
    if concat_multi_camera == "robotwin":
        if num_cameras != 3:
            raise ValueError(
                f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
            )
        cam_top = transforms_F.resize(
            video[0],
            size=[256, 320],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )  # [T_video, C, 256, 320]
        cam_left = transforms_F.resize(
            video[1],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )  # [T_video, C, 128, 160]
        cam_right = transforms_F.resize(
            video[2],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )  # [T_video, C, 128, 160]
        bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
        return torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
    if num_cameras > 1:
        if concat_multi_camera == "horizontal":
            return torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
        if concat_multi_camera == "vertical":
            return torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
        raise ValueError(
            f"Invalid concat_multi_camera: {concat_multi_camera}. "
            "Expected one of: horizontal, vertical, robotwin."
        )
    return video.squeeze(0)  # [T_video, C, H, W]


def to_model_frames(video: torch.Tensor, concat_multi_camera: Optional[str], video_size) -> torch.Tensor:
    """``[num_cameras, T, C, H, W]`` frames in [0, 1] -> the model's ``[T, C, H, W]`` input in [-1, 1]."""
    return canvas_to_model_frames(compose_multi_camera(video, concat_multi_camera), video_size)


def canvas_to_model_frames(video: torch.Tensor, video_size) -> torch.Tensor:
    """A composed ``[T, C, H, W]`` canvas in [0, 1] -> the model's ``[T, C, H, W]`` input in [-1, 1]."""
    # final resize and normalization
    video = ResizeSmallestSideAspectPreserving(args={"img_w": video_size[1], "img_h": video_size[0]})(video)
    video = CenterCrop(args={"img_w": video_size[1], "img_h": video_size[0]})(video)
    return Normalize(args={"mean": 0.5, "std": 0.5})(video)  # [T_video, C, H, W]


class RobotVideoDataset(torch.utils.data.Dataset):
    base_dataset_cls = BaseLerobotDataset

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        use_text_embed_cache=True,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        tolerance_s: Optional[float] = None,
        video_backend: Optional[str] = None,
        color_jitter: Optional[dict] = None,
    ):
        """``color_jitter`` (e.g. ``{brightness: 0.3, contrast: 0.3, saturation: 0.3}``) is torchvision's
        ``ColorJitter`` applied to the composed multi-camera canvas, one draw per sample shared by every
        video frame and camera of it, before the final resize and normalization. It is a training
        augmentation: a dataset with ``is_training_set=False`` ignores it, and inference servers build
        the canvas with ``to_model_frames``, which never jitters."""
        self.lerobot_dataset = self.base_dataset_cls(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            image_subsample_stride=action_video_freq_ratio,
            tolerance_s=tolerance_s,
            video_backend=video_backend,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.use_text_embed_cache = bool(use_text_embed_cache)
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.color_jitter = (
            transforms_v2.ColorJitter(**(OmegaConf.to_container(color_jitter) if isinstance(color_jitter, DictConfig) else color_jitter))
            if color_jitter and is_training_set
            else None
        )

        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if self.lerobot_dataset.presample_images:
                processor.num_image_steps = len(self.video_sample_indices)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def to_model_frames(self, video: torch.Tensor) -> torch.Tensor:
        """``[num_cameras, T, C, H, W]`` frames in [0, 1] -> the model's ``[T, C, H, W]`` input in [-1, 1]."""
        return to_model_frames(video, self.concat_multi_camera, self.video_size)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            if not self.lerobot_dataset.presample_images:
                video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            if not self.lerobot_dataset.presample_images:
                video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        if not self.lerobot_dataset.presample_images:
            image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        canvas = compose_multi_camera(video, self.concat_multi_camera)  # [T_video, C, H', W'] in [0, 1]
        if self.color_jitter is not None:
            canvas = self.color_jitter(canvas)  # one draw for the whole canvas, every frame
        video = canvas_to_model_frames(canvas, self.video_size)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.use_text_embed_cache:
            context, context_mask = self._get_cached_text_context(instruction)
            # NOTE: to keep consistent with wan2.2's behavior
            context[~context_mask] = 0.0
            data["context"] = context
            data["context_mask"] = torch.ones_like(context_mask)
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
