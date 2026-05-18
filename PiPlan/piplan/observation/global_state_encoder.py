from __future__ import annotations

import numpy as np

from piplan.core.port import Port
from piplan.core.state import WorldState
from piplan.core.task import TaskStatus, TaskType
from .pi05_image_adapter import Pi05ImageAdapter
from .task_prompt_builder import TaskPromptBuilder


class GlobalStateEncoder:
    """Encode WorldState into pi0.5 centralized observation structure.

    Output:
      {
        "map_features": float32 array [3] containing width, height, agent count.
        "agent_tokens": float32 array [N, 16],
        "map_image": float32 array [3, 224, 224], normalized to [-1, 1],
        "task_text": str,
        "agent_order": list[int],
        "metadata": dict
      }
    Units intentionally match Dorabot `joint_bc_v1`: raw map meters,
    raw velocity, angle, relative destination vector, and task-aware codes.
    """

    schema_version = "piplan_pi05_observation_v1"
    token_dim = 16

    def __init__(self, max_speed: float = 1.5, image_size: int = 224):
        self.max_speed = max(max_speed, 1e-6)
        self.image_adapter = Pi05ImageAdapter(size=image_size)
        self.prompt_builder = TaskPromptBuilder()

    def encode(self, world: WorldState) -> dict:
        render_state = self.render_state(world)
        tokens = [self._agent_token(agent, world) for agent in sorted(world.agents, key=lambda item: item.id)]
        return {
            "schema_version": self.schema_version,
            "map_features": np.asarray(
                [float(world.warehouse.width), float(world.warehouse.height), float(len(world.agents))],
                dtype="float32",
            ),
            "agent_tokens": np.asarray(tokens, dtype="float32").reshape((len(tokens), self.token_dim)),
            "map_image": self.image_adapter.to_joint_image(render_state),
            "task_text": self.prompt_builder.build(render_state),
            "agent_order": world.agent_order(),
            "metadata": {
                "tick": world.step,
                "time": world.time,
                "render_state": render_state,
                "tasks": [self._task(task) for task in world.tasks],
                "ports": [self._port(port) for port in world.all_ports()],
            },
        }

    def render_state(self, world: WorldState) -> dict:
        return {
            "schema_version": "piplan_render_state_v1",
            "step": world.step,
            "time": world.time,
            "map": {"width": world.warehouse.width, "height": world.warehouse.height},
            "agents": [self._agent_render(agent) for agent in sorted(world.agents, key=lambda item: item.id)],
            "loading_ports": [self._port(port) for port in world.warehouse.loading_ports],
            "unloading_ports": [self._port(port) for port in world.warehouse.unloading_ports],
            "obstacles": [
                {"id": idx, "x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height}
                for idx, rect in enumerate(world.warehouse.obstacles)
            ],
        }

    def _agent_token(self, agent, world: WorldState) -> list[float]:
        destination = agent.destination
        if destination is None:
            goal_x = 0.0
            goal_y = 0.0
            distance = 0.0
            has_destination = 0.0
        else:
            goal_x = destination.x - agent.position.x
            goal_y = destination.y - agent.position.y
            distance = (goal_x ** 2 + goal_y ** 2) ** 0.5
            has_destination = 1.0
        task = agent.task
        return [
            float(agent.id),
            float(agent.position.x),
            float(agent.position.y),
            float(agent.velocity.vx),
            float(agent.velocity.vy),
            float(agent.angle),
            float(goal_x),
            float(goal_y),
            float(distance),
            float(has_destination),
            float(agent.ray_min or 0.0),
            self._task_code(task.task_type),
            self._port_kind_code(task.port_kind),
            1.0 if task.has_item else 0.0,
            self._encoded_port_id(world, task.port_id, task.port_kind),
            self._encoded_destination_port_id(world, task),
        ]

    def _task_status_one_hot(self, agent) -> list[float]:
        status = agent.task.status
        if status == TaskStatus.PENDING or agent.assigned_task_id is None:
            return [1.0, 0.0, 0.0, 0.0]
        if status == TaskStatus.ASSIGNED:
            return [0.0, 1.0, 0.0, 0.0]
        if status == TaskStatus.IN_PROGRESS:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]

    def _port_type_one_hot(self, agent) -> list[float]:
        if agent.task.port_kind == "input":
            return [1.0, 0.0]
        if agent.task.port_kind == "output":
            return [0.0, 1.0]
        return [0.0, 0.0]

    def _task_code(self, task_type: TaskType | str | None) -> float:
        value = task_type.value if isinstance(task_type, TaskType) else task_type
        if value == TaskType.GO_TO_LOADING_PORT.value:
            return 1.0
        if value == TaskType.GO_TO_UNLOADING_PORT.value:
            return 2.0
        if value in {None, TaskType.IDLE.value}:
            return 0.0
        return -1.0

    def _port_kind_code(self, port_kind: str | None) -> float:
        if port_kind == "input":
            return 1.0
        if port_kind == "output":
            return 2.0
        return 0.0

    def _encoded_port_id(self, world: WorldState, port_id: int | None, port_kind: str | None) -> float:
        if port_id is None:
            return -1.0
        ports = world.warehouse.loading_ports if port_kind == "input" else world.warehouse.unloading_ports
        for idx, port in enumerate(ports, start=1):
            if port.id == port_id:
                return float(idx)
        return float(port_id)

    def _encoded_destination_port_id(self, world: WorldState, task) -> float:
        if not task.has_item or task.destination_port_id is None:
            return -1.0
        for idx, port in enumerate(world.warehouse.unloading_ports, start=1):
            if port.id == task.destination_port_id:
                return float(idx)
        return float(task.destination_port_id)

    def _agent_render(self, agent) -> dict:
        destination = agent.destination
        task = agent.task
        return {
            "id": agent.id,
            "x": agent.position.x,
            "y": agent.position.y,
            "vx": agent.velocity.vx,
            "vy": agent.velocity.vy,
            "angle": agent.angle,
            "radius": agent.radius,
            "state": agent.state,
            "ray_min": agent.ray_min,
            "destination": None if destination is None else {"x": destination.x, "y": destination.y},
            "task": {
                "type": task.task_type.value,
                "status": task.status.value,
                "port_id": task.port_id,
                "port_kind": task.port_kind,
                "has_item": task.has_item,
            },
        }

    def _port(self, port: Port) -> dict:
        return {
            "id": port.id,
            "kind": port.port_type.value,
            "x": port.rect.x,
            "y": port.rect.y,
            "width": port.rect.width,
            "height": port.rect.height,
            "capacity": port.capacity,
            "assigned_agent_ids": list(port.assigned_agent_ids),
            "num_items": port.num_items,
        }

    def _task(self, task) -> dict:
        return {
            "id": task.id,
            "status": task.status.value,
            "assigned_agent_id": task.assigned_agent_id,
            "source_port_id": task.source_port_id,
            "destination_port_id": task.destination_port_id,
            "assign_time": task.assign_time,
            "done_time": task.done_time,
        }
