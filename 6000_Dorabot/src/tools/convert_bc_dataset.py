#!/usr/bin/env python
"""Convert Dorabot bc_v1 JSONL logs into training datasets.

Examples:
  python tools/convert_bc_dataset.py \
    --input data_samples/bc_expert.jsonl \
    --output converted/dorabot_bc_flat \
    --format flat

  python tools/convert_bc_dataset.py \
    --input data_samples/bc_expert.jsonl \
    --output converted/dorabot_lerobot \
    --format lerobot \
    --repo_id local/dorabot-pi05-bc \
    --include_images
"""

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geometry import Point  # noqa: E402
from main_planners.pi05_image_adapter import Pi05ImageAdapter  # noqa: E402


ACTION_SOURCES = {
    "expert": "expert_velocity",
    "executed": "executed_velocity",
    "safe": "safe_velocity",
    "proposed": "proposed_velocity",
}

JOINT_ACTION_SOURCES = {
    "expert": "joint_expert_action",
    "executed": "joint_executed_action",
    "safe": "joint_safe_action",
    "proposed": "joint_proposed_action",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Dorabot bc_v1 JSONL logs.")
    parser.add_argument("--input", action="append", required=True, help="Input JSONL path or glob. Can be repeated.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--format", choices=["flat", "lerobot"], default="flat")
    parser.add_argument("--repo_id", default="local/dorabot-pi05-bc", help="LeRobot repo id for --format lerobot.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action_source", choices=sorted(ACTION_SOURCES), default="expert")
    parser.add_argument("--include_images", action="store_true", help="Add top-down image observations when possible.")
    parser.add_argument("--image_key", default="observation.images.topdown")
    parser.add_argument("--pad_state_dim", type=int, default=0, help="Pad/truncate observation.state to this dim. 0 keeps inferred dim.")
    parser.add_argument("--pad_action_dim", type=int, default=0, help="Pad/truncate action to this dim. 0 keeps inferred dim.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_videos", action="store_true", help="Store LeRobot images as videos instead of image files.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise SystemExit("No bc_v1 rows found.")

    action_key = ACTION_SOURCES[args.action_source]
    examples = [row_to_example(row, action_key, args.include_images) for row in rows]
    examples = [example for example in examples if example is not None]
    if not examples:
        raise SystemExit(f"No rows had action source `{action_key}`.")

    state_dim = args.pad_state_dim or max(len(example["state"]) for example in examples)
    action_dim = args.pad_action_dim or max(len(example["action"]) for example in examples)
    for example in examples:
        example["state"] = pad(example["state"], state_dim)
        example["action"] = pad(example["action"], action_dim)

    output = Path(args.output)
    prepare_output(output, args.overwrite, create_dir=args.format == "flat")
    if args.format == "flat":
        write_flat(output, examples, args, state_dim, action_dim)
    else:
        write_lerobot(output, examples, args, state_dim, action_dim)

    print(f"Converted {len(examples)} frames from {len(rows)} rows.")
    print(f"Output: {output}")
    print(f"state_dim={state_dim} action_dim={action_dim} include_images={args.include_images}")


def load_rows(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])

    rows = []
    for path in paths:
        with open(path) as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") not in ["bc_v1", "joint_bc_v1"]:
                    continue
                row["_source_file"] = path
                row["_source_line"] = line_no
                rows.append(row)
    return rows


def row_to_example(row, action_key, include_image):
    if row.get("schema_version") == "joint_bc_v1":
        return joint_row_to_example(row, JOINT_ACTION_SOURCES[action_key_from_value(action_key)], include_image)

    action = row.get("action", {}).get(action_key)
    if action is None:
        return None

    observation = row.get("observation", {})
    language = row.get("language", {})
    example = {
        "source_file": row.get("_source_file"),
        "source_line": row.get("_source_line"),
        "episode_key": episode_key(row),
        "time": row.get("time"),
        "agent_id": row.get("agent_id"),
        "state": list(observation.get("state_vector", [])),
        "action": list(action),
        "task": language.get("task_text") or "navigate in warehouse",
        "metadata": {
            "planner": row.get("planner", {}),
            "target": row.get("target", {}),
            "shield_reason": row.get("action", {}).get("shield_reason"),
        },
    }
    if include_image:
        example["image"] = make_image(row)
    return example


def action_key_from_value(value):
    for key, candidate in ACTION_SOURCES.items():
        if candidate == value:
            return key
    return "expert"


def joint_row_to_example(row, action_key, include_image):
    action = row.get("action", {}).get(action_key)
    if action is None:
        return None
    observation = row.get("observation", {})
    language = row.get("language", {})
    example = {
        "source_file": row.get("_source_file"),
        "source_line": row.get("_source_line"),
        "episode_key": row.get("_source_file", "unknown") + "::joint",
        "time": row.get("time"),
        "agent_id": None,
        "state": list(observation.get("joint_state_vector", [])),
        "action": list(action),
        "task": language.get("task_text") or "plan coordinated warehouse motion",
        "metadata": {
            "planner": row.get("planner", {}),
            "agent_actions": row.get("action", {}).get("agents", []),
        },
    }
    if include_image:
        example["image"] = make_joint_image(row)
    return example


def make_joint_image(row):
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
    return Pi05ImageAdapter().to_joint_image(global_state)


def episode_key(row):
    return "{}::agent_{}".format(row.get("_source_file", "unknown"), row.get("agent_id", 0))


def make_image(row):
    global_state = row.get("global_state")
    if not global_state:
        global_state = minimal_global_state(row)
    ego_state = row.get("observation", {}).get("ego", {})
    target = row.get("target", {}).get("destination")
    ego = SimpleNamespace(
        id=row.get("agent_id", 0),
        position=Point(ego_state.get("x", 0.0), ego_state.get("y", 0.0)),
        destination_location=None if target is None else Point(target["x"], target["y"]),
    )
    return Pi05ImageAdapter().to_image(ego, global_state)


def minimal_global_state(row):
    observation = row.get("observation", {})
    ego = observation.get("ego", {})
    agent_id = row.get("agent_id", 0)
    return {
        "ego_agent_id": agent_id,
        "map": observation.get("map", {}),
        "agents": [{
            "id": agent_id,
            "is_ego": True,
            "x": ego.get("x", 0.0),
            "y": ego.get("y", 0.0),
            "vx": ego.get("vx", 0.0),
            "vy": ego.get("vy", 0.0),
            "angle": ego.get("angle", 0.0),
        }] + observation.get("nearby_agents", []),
        "loading_ports": [],
        "unloading_ports": [],
        "obstacles": [],
    }


def pad(values, dim):
    result = [float(value) for value in values[:dim]]
    if len(result) < dim:
        result.extend([0.0] * (dim - len(result)))
    return result


def prepare_output(output, overwrite, create_dir=True):
    if output.exists():
        if not overwrite:
            raise SystemExit(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    if create_dir:
        output.mkdir(parents=True, exist_ok=True)


def write_flat(output, examples, args, state_dim, action_dim):
    meta = {
        "format": "dorabot_flat_bc_v1",
        "num_frames": len(examples),
        "num_episodes": len(group_examples(examples)),
        "fps": args.fps,
        "action_source": args.action_source,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "include_images": args.include_images,
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    image_dir = output / "images"
    if args.include_images:
        image_dir.mkdir()

    with open(output / "data.jsonl", "w") as handle:
        for idx, example in enumerate(examples):
            row = {
                "episode_key": example["episode_key"],
                "time": example["time"],
                "agent_id": example["agent_id"],
                "observation.state": example["state"],
                "action": example["action"],
                "task": example["task"],
                "metadata": example["metadata"],
            }
            if args.include_images:
                path = image_dir / f"frame_{idx:08d}.npy"
                np.save(path, example["image"])
                row[args.image_key] = str(path.relative_to(output))
            handle.write(json.dumps(row) + "\n")


def write_lerobot(output, examples, args, state_dim, action_dim):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION, OBS_STATE

    features = {
        OBS_STATE: {"dtype": "float32", "shape": (state_dim,), "names": None},
        ACTION: {"dtype": "float32", "shape": (action_dim,), "names": ["vx", "vy"] if action_dim == 2 else None},
    }
    if args.include_images:
        features[args.image_key] = {
            "dtype": "video" if args.use_videos else "image",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=output,
        robot_type="dorabot_sim",
        use_videos=args.use_videos,
        image_writer_threads=2 if args.include_images else 0,
    )

    for _, episode_examples in group_examples(examples).items():
        for example in episode_examples:
            frame = {
                OBS_STATE: torch.tensor(example["state"], dtype=torch.float32),
                ACTION: torch.tensor(example["action"], dtype=torch.float32),
                "task": example["task"],
            }
            if args.include_images:
                frame[args.image_key] = example["image"]
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()


def group_examples(examples):
    groups = {}
    for example in examples:
        groups.setdefault(example["episode_key"], []).append(example)
    return groups


if __name__ == "__main__":
    main()
