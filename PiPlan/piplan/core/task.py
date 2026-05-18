from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .geometry import Point


class TaskType(str, Enum):
    IDLE = "idle"
    GO_TO_LOADING_PORT = "go_to_loading_port"
    GO_TO_UNLOADING_PORT = "go_to_unloading_port"
    NAVIGATE = "navigate"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Task:
    """Task lifecycle state.

    Times are simulation seconds. A task starts pending, gets assigned to an
    idle agent, becomes in-progress while the agent carries it from input to
    output, and completes when the output port is reached.
    """

    id: int = -1
    task_type: TaskType = TaskType.IDLE
    status: TaskStatus = TaskStatus.PENDING
    destination: Point | None = None
    source: Point | None = None
    port_id: int | None = None
    port_kind: str | None = None
    source_port_id: int | None = None
    destination_port_id: int | None = None
    assigned_agent_id: int | None = None
    has_item: bool = False
    created_time: float = 0.0
    assign_time: float | None = None
    start_time: float | None = None
    done_time: float | None = None

    @property
    def is_active(self) -> bool:
        return self.destination is not None and self.status != TaskStatus.COMPLETED


class TaskManager:
    """Assigns input-to-output warehouse tasks and computes throughput.

    Input: mutable agents, ports, and task list from WorldState.
    Output: task status updates plus `completed_task_ids_last_tick`.
    """

    def __init__(self, arrival_radius: float = 0.45):
        self.arrival_radius = arrival_radius
        self.completed_task_ids_last_tick: list[int] = []

    def step(self, world_state) -> list[int]:
        self.completed_task_ids_last_tick = []
        self._assign_pending_tasks(world_state)
        self._advance_assigned_tasks(world_state)
        return list(self.completed_task_ids_last_tick)

    def pph(self, world_state) -> float:
        if world_state.time <= 0.0:
            return 0.0
        completed = sum(1 for task in world_state.tasks if task.status == TaskStatus.COMPLETED)
        return completed / (world_state.time / 3600.0)

    def _assign_pending_tasks(self, world_state) -> None:
        idle_agents = [
            agent for agent in sorted(world_state.agents, key=lambda item: item.id)
            if agent.assigned_task_id is None
        ]
        pending_tasks = [
            task for task in sorted(world_state.tasks, key=lambda item: item.id)
            if task.status == TaskStatus.PENDING
        ]
        for agent, task in zip(idle_agents, pending_tasks):
            input_port = world_state.get_port(task.source_port_id)
            if input_port is not None and not input_port.assign(agent.id):
                continue
            task.status = TaskStatus.ASSIGNED
            task.assigned_agent_id = agent.id
            task.assign_time = world_state.time
            task.task_type = TaskType.GO_TO_LOADING_PORT
            task.destination = task.source
            task.port_id = task.source_port_id
            task.port_kind = "input"
            agent.assigned_task_id = task.id
            agent.task = task
            agent.state = "assigned"

    def _advance_assigned_tasks(self, world_state) -> None:
        for agent in world_state.agents:
            if agent.assigned_task_id is None:
                continue
            task = world_state.get_task(agent.assigned_task_id)
            if task is None or task.destination is None:
                continue
            if agent.position.distance_to(task.destination) > self.arrival_radius:
                continue
            if task.status == TaskStatus.ASSIGNED:
                input_port = world_state.get_port(task.source_port_id)
                if input_port is not None:
                    input_port.release(agent.id)
                output_port = world_state.get_port(task.destination_port_id)
                if output_port is not None and not output_port.assign(agent.id):
                    continue
                task.status = TaskStatus.IN_PROGRESS
                task.task_type = TaskType.GO_TO_UNLOADING_PORT
                task.has_item = True
                task.start_time = world_state.time
                task.destination = world_state.port_point(task.destination_port_id)
                task.port_id = task.destination_port_id
                task.port_kind = "output"
                agent.task = task
                agent.state = "carrying"
            elif task.status == TaskStatus.IN_PROGRESS:
                output_port = world_state.get_port(task.destination_port_id)
                if output_port is not None:
                    output_port.release(agent.id)
                task.status = TaskStatus.COMPLETED
                task.done_time = world_state.time
                task.has_item = False
                agent.assigned_task_id = None
                agent.task = Task(status=TaskStatus.COMPLETED)
                agent.state = "done"
                self.completed_task_ids_last_tick.append(task.id)
