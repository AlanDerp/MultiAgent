from __future__ import annotations

from types import SimpleNamespace

from piplan.observation.pi05_image_adapter import Pi05ImageAdapter


class ExpertDatasetConverter:
    """Convert Dorabot bc_v1/joint_bc_v1 logs into PiPlan pi0.5 examples."""

    action_sources = {
        "expert": ("expert_velocity", "joint_expert_action"),
        "executed": ("executed_velocity", "joint_executed_action"),
        "safe": ("safe_velocity", "joint_safe_action"),
        "proposed": ("proposed_velocity", "joint_proposed_action"),
    }

    def __init__(self, include_images: bool = True, action_source: str = "expert"):
        self.include_images = include_images
        self.action_source = action_source
        self.image_adapter = Pi05ImageAdapter()

    def convert(self, rows: list[dict]) -> list[dict]:
        examples = []
        for row in rows:
            example = self._convert_row(row)
            if example:
                examples.append(example)
        return examples

    def _convert_row(self, row: dict) -> dict | None:
        per_agent_key, joint_key = self.action_sources[self.action_source]
        if row.get("schema_version") == "joint_bc_v1":
            action = row.get("action", {}).get(joint_key)
            if action is None:
                return None
            observation = row.get("observation", {})
            global_state = row.get("global_state") or {
                "map": observation.get("map", {}),
                "agents": observation.get("agents", []),
                "loading_ports": [],
                "unloading_ports": [],
                "obstacles": [],
            }
            example = {
                "state": list(observation.get("joint_state_vector", [])),
                "action": list(action),
                "task": row.get("language", {}).get("task_text") or "plan coordinated warehouse motion",
                "metadata": {"source": row.get("_source_file"), "line": row.get("_source_line"), "legacy_schema": "joint_bc_v1"},
            }
            if self.include_images:
                example["image"] = self.image_adapter.to_joint_image(global_state)
            return example

        action = row.get("action", {}).get(per_agent_key)
        if action is None:
            return None
        observation = row.get("observation", {})
        global_state = row.get("global_state") or self._minimal_global_state(row)
        example = {
            "state": list(observation.get("state_vector", [])),
            "action": list(action),
            "task": row.get("language", {}).get("task_text") or "navigate in warehouse",
            "metadata": {"source": row.get("_source_file"), "line": row.get("_source_line"), "legacy_schema": "bc_v1"},
        }
        if self.include_images:
            example["image"] = self._single_agent_image(row, global_state)
        return example

    def _single_agent_image(self, row: dict, global_state: dict):
        from piplan.core.geometry import Point

        ego_state = row.get("observation", {}).get("ego", {})
        target = row.get("target", {}).get("destination")
        ego = SimpleNamespace(
            id=row.get("agent_id", 0),
            position=Point(ego_state.get("x", 0.0), ego_state.get("y", 0.0)),
            destination_location=None if target is None else Point(target["x"], target["y"]),
        )
        single_adapter = _SingleAgentImageAdapter(self.image_adapter)
        return single_adapter.to_image(ego, global_state)

    def _minimal_global_state(self, row: dict) -> dict:
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


class _SingleAgentImageAdapter:
    def __init__(self, joint_adapter: Pi05ImageAdapter):
        self.joint_adapter = joint_adapter

    def to_image(self, ego, global_state):
        if ego.destination_location is not None:
            for agent in global_state.get("agents", []):
                if agent.get("id") == ego.id:
                    agent["destination"] = {"x": ego.destination_location.x, "y": ego.destination_location.y}
        return self.joint_adapter.to_joint_image(global_state)
