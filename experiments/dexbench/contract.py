"""The DexBench RBY1-Wuji2 contract the GA wrist-camera recipe trains on and the server states at ``/info``.

The 62 policy joints exclude the four wheels; the order is the DexBench 62-joint LeRobot layout's
(``meta/info.json`` ``features.action.names``). ``observation.state`` lists the same joints with a
``_Pos`` suffix. Run this file on a dataset root to check it before training:

    python experiments/dexbench/contract.py /path/to/gr00t_62d_320x180_4cam
"""

import json
import sys
from pathlib import Path

EMBODIMENT = "dexbench_rby1_wuji2"
FPS = 30
# Image order is the canvas order: top slot, then the left and right bottom slots.
CAMERAS = ("camera_ego_left", "camera_wrist_left", "camera_wrist_right")
FRAME_HW = (180, 320)

_ARMS = [f"{side}_arm_{i}" for side in ("left", "right") for i in range(7)]
_HEAD = ["head_0", "head_1"]
_FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
_STAGES = (
    ("cmc_flex", "mcp_flex"),
    ("cmc_abd", "mcp_abd"),
    ("mcp", "pip"),
    ("ip", "dip"),
)
# For each stage, the five left-hand joints, then the five right-hand joints.
_HANDS = [
    f"{side}_{finger}_{thumb_stage if finger == 'thumb' else finger_stage}"
    for thumb_stage, finger_stage in _STAGES
    for side in ("left", "right")
    for finger in _FINGERS
]
_TORSO = [f"torso_{i}" for i in range(6)]
JOINT_NAMES = tuple(_ARMS + _HEAD + _HANDS + _TORSO)
assert len(JOINT_NAMES) == 62 and len(set(JOINT_NAMES)) == 62


def dataset_problems(root: Path) -> list[str]:
    """Why a LeRobot 2.1 root is not a 62-joint, four-camera 320x180 DexBench set, or [] when it is."""
    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    features = info["features"]
    problems = []
    if info.get("codebase_version") != "v2.1":
        problems.append(f"codebase_version {info.get('codebase_version')}, expected v2.1")
    if int(info.get("fps", 0)) != FPS:
        problems.append(f"fps {info.get('fps')}, expected {FPS}")
    if list(features.get("action", {}).get("names") or []) != list(JOINT_NAMES):
        problems.append("action names are not the 62 DexBench joints in order")
    state = [str(n).removesuffix("_Pos") for n in features.get("observation.state", {}).get("names") or []]
    if state != list(JOINT_NAMES):
        problems.append("observation.state names are not the 62 DexBench joints in order")
    for camera in CAMERAS:
        feature = features.get(f"observation.images.{camera}")
        if feature is None:
            problems.append(f"missing observation.images.{camera}")
        elif list(feature.get("shape", [])[1:]) != list(FRAME_HW):
            problems.append(f"{camera} frames are {feature.get('shape')}, expected [3, {FRAME_HW[0]}, {FRAME_HW[1]}]")
    return problems


if __name__ == "__main__":
    failed = False
    for arg in sys.argv[1:]:
        problems = dataset_problems(Path(arg))
        print(f"{arg}: {'ok' if not problems else '; '.join(problems)}")
        failed |= bool(problems)
    sys.exit(1 if failed or len(sys.argv) < 2 else 0)
