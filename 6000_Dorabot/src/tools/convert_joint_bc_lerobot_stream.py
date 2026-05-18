#!/usr/bin/env python3
"""Stream-convert large joint_bc_v1 JSONL logs into a LeRobot dataset."""

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from main_planners.pi05_image_adapter import Pi05ImageAdapter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Stream-convert joint_bc_v1 JSONL to LeRobot.")
    parser.add_argument("--input", action="append", required=True, help="Input JSONL path or glob.")
    parser.add_argument("--output", required=True, help="Output LeRobot dataset root.")
    parser.add_argument("--repo_id", default="local/dorabot-pi05-joint-taskaware-quick")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pad_state_dim", type=int, default=163)
    parser.add_argument("--pad_action_dim", type=int, default=20)
    parser.add_argument("--image_key", default="observation.images.topdown")
    parser.add_argument("--include_images", action="store_true")
    parser.add_argument("--use_videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10000)
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    return paths


def pad(values, dim):
    result = [float(value) for value in values[:dim]]
    if len(result) < dim:
        result.extend([0.0] * (dim - len(result)))
    return result


def make_joint_image(row, image_adapter):
    global_state = row.get("global_state")
    if not global_state:
        observation = row.get("observation", {})
        global_state = {
            "map": observation.get("map", {}),
            "agents": observation.get("agents", []),
            "loading_ports": [],
            "unloading_ports": [],
            "obstacles": [],
        }
    return image_adapter.to_joint_image(global_state)


def create_dataset(args):
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    features = {
        OBS_STATE: {"dtype": "float32", "shape": (args.pad_state_dim,), "names": None},
        ACTION: {"dtype": "float32", "shape": (args.pad_action_dim,), "names": None},
    }
    if args.include_images:
        features[args.image_key] = {
            "dtype": "video" if args.use_videos else "image",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=output,
        robot_type="dorabot_sim",
        use_videos=args.use_videos,
        image_writer_threads=4 if args.include_images else 0,
    )


def should_start_new_episode(last_step, last_time, row):
    step = row.get("step")
    row_time = row.get("time")
    if last_step is not None and step is not None and step <= last_step:
        return True
    if last_time is not None and row_time is not None and row_time < last_time:
        return True
    return False


def main():
    args = parse_args()
    dataset = create_dataset(args)
    image_adapter = Pi05ImageAdapter() if args.include_images else None
    total_frames = 0
    total_episodes = 0
    frames_in_episode = 0
    last_step = None
    last_time = None

    for path in expand_inputs(args.input):
        with open(path) as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != "joint_bc_v1":
                    continue
                if frames_in_episode and should_start_new_episode(last_step, last_time, row):
                    dataset.save_episode()
                    total_episodes += 1
                    frames_in_episode = 0

                observation = row.get("observation", {})
                action = row.get("action", {}).get("joint_expert_action")
                if action is None:
                    continue
                frame = {
                    OBS_STATE: torch.tensor(
                        pad(observation.get("joint_state_vector", []), args.pad_state_dim),
                        dtype=torch.float32,
                    ),
                    ACTION: torch.tensor(pad(action, args.pad_action_dim), dtype=torch.float32),
                    "task": row.get("language", {}).get("task_text") or "plan coordinated warehouse motion",
                }
                if args.include_images:
                    frame[args.image_key] = make_joint_image(row, image_adapter)
                dataset.add_frame(frame)

                total_frames += 1
                frames_in_episode += 1
                last_step = row.get("step")
                last_time = row.get("time")
                if args.progress_interval > 0 and total_frames % args.progress_interval == 0:
                    print(
                        f"converted frames={total_frames} episodes_saved={total_episodes} "
                        f"current_episode_frames={frames_in_episode} source={path}:{line_no}",
                        flush=True,
                    )

    if frames_in_episode:
        dataset.save_episode()
        total_episodes += 1
    dataset.finalize()
    print(f"Converted {total_frames} frames across {total_episodes} episodes.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
