from __future__ import annotations

from pathlib import Path

from piplan.data.schemas import pad


class LeRobotDatasetWriter:
    """Write PiPlan joint rollouts as LeRobot datasets.

    Inputs are frame dictionaries with state/action/task/image plus optional
    reward and done. Output uses LeRobot parquet + meta layout.
    """

    def write(
        self,
        output: str | Path,
        examples: list[dict],
        repo_id: str,
        fps: int = 30,
        include_images: bool = True,
        append: bool = False,
    ):
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import ACTION, OBS_STATE

        output = Path(output)
        if append and not output.exists():
            append = False
        state_dim = max(len(example["state"]) for example in examples)
        action_dim = max(len(example["action"]) for example in examples)
        features = {
            OBS_STATE: {"dtype": "float32", "shape": (state_dim,), "names": None},
            ACTION: {"dtype": "float32", "shape": (action_dim,), "names": None},
            "reward": {"dtype": "float32", "shape": (1,), "names": None},
            "done": {"dtype": "bool", "shape": (1,), "names": None},
        }
        if include_images:
            features["observation.images.topdown"] = {
                "dtype": "image",
                "shape": (3, 224, 224),
                "names": ["channels", "height", "width"],
            }

        if append:
            dataset = LeRobotDataset(repo_id=repo_id, root=output)
        else:
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                root=output,
                robot_type="piplan_dorabot_sim",
                use_videos=False,
                image_writer_threads=2 if include_images else 0,
            )
        for example in examples:
            frame = {
                OBS_STATE: torch.tensor(pad(example["state"], state_dim), dtype=torch.float32),
                ACTION: torch.tensor(pad(example["action"], action_dim), dtype=torch.float32),
                "reward": torch.tensor([float(example.get("reward", 0.0))], dtype=torch.float32),
                "done": bool(example.get("done", False)),
                "task": example["task"],
            }
            if include_images:
                frame["observation.images.topdown"] = example["image"]
            dataset.add_frame(frame)
        dataset.save_episode()
        dataset.finalize()
