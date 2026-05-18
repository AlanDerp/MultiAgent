from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from piplan.data.schemas import PIPLAN_ROLLOUT_SCHEMA


class JsonlRolloutLogger:
    """Tick-level rollout logger for LeRobot-compatible collection.

    Per tick record:
      observation.state: flat float list, length N*15.
      observation.image_path: relative `.npy` path for [3,224,224] image.
      action: flat joint action [vx0, vy0, ..., vxN, vyN].
      reward: immediate scalar reward.
      done: episode termination flag.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.path.parent / (self.path.stem + "_images")
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.frame_index = 0

    def log(self, simulator, done: bool | None = None) -> dict:
        world = simulator._state()
        obs = simulator.last_observation
        if obs is None:
            obs = simulator.observation_builder.build(world)
        image_path = self._save_image(obs.map_image)
        record = {
            "schema_version": PIPLAN_ROLLOUT_SCHEMA,
            "step": world.step,
            "time": world.time,
            "observation": {
                "state": obs.state_vector,
                "image_path": str(image_path.relative_to(self.path.parent)),
                "agent_order": obs.agent_order,
                "task_text": obs.task_text,
            },
            "action": self._joint_action(obs.agent_order, simulator.last_action_log),
            "reward": self.reward(simulator),
            "done": self.done(world) if done is None else done,
            "metadata": {
                "completed_task_ids": list(world.completed_task_ids_last_tick),
                "safety_events": list(getattr(simulator.actuator.safety, "last_tick_events", [])),
            },
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    def reward(self, simulator) -> float:
        world = simulator._state()
        completed = len(world.completed_task_ids_last_tick)
        safety_collisions = len(getattr(simulator.actuator.safety, "last_tick_events", []))
        stalled_index = world.stalled_agent_ticks_last_tick / max(len(world.agents), 1)
        return 1.0 * completed - 0.5 * safety_collisions - 0.01 * stalled_index

    def done(self, world) -> bool:
        return bool(world.tasks) and all(task.status.value == "completed" for task in world.tasks)

    def _save_image(self, image: np.ndarray) -> Path:
        path = self.image_dir / f"frame_{self.frame_index:08d}.npy"
        np.save(path, image)
        self.frame_index += 1
        return path

    def _joint_action(self, agent_order: list[int], action_log) -> list[float]:
        result = []
        applied = action_log.applied if action_log else {}
        for agent_id in agent_order:
            action = applied.get(agent_id)
            if action is None:
                result.extend([0.0, 0.0])
            else:
                result.extend([action.vx, action.vy])
        return result
