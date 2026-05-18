from __future__ import annotations

from dataclasses import dataclass, field

from .action import VelocityAction
from .geometry import Point
from .task import Task, TaskStatus


@dataclass
class Agent:
    """Runtime agent state.

    Units: position in meters, velocity in meters/second, angle in radians.
    Sensor fields are normalized proximity readings used by observation tokens.
    """

    id: int
    position: Point
    angle: float = 0.0
    velocity: VelocityAction = field(default_factory=lambda: VelocityAction(0.0, 0.0))
    cruise_speed: float = 1.5
    radius: float = 0.35
    task: Task = field(default_factory=Task)
    assigned_task_id: int | None = None
    ray_min: float | None = None
    ray_min_front: float = 1.0
    ray_min_left: float = 1.0
    ray_min_right: float = 1.0
    state: str = "idle"

    @property
    def destination(self) -> Point | None:
        return self.task.destination

    def has_destination(self) -> bool:
        return self.destination is not None

    def task_status(self) -> TaskStatus:
        if self.task.status == TaskStatus.IN_PROGRESS and self.task.has_item:
            return TaskStatus.IN_PROGRESS
        return self.task.status

    def step(self, dt: float) -> None:
        self.position = self.position.moved(self.velocity.vx, self.velocity.vy, dt)
