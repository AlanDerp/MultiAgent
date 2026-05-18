from __future__ import annotations

from dataclasses import dataclass, field

from .agent import Agent
from .geometry import Point
from .port import Port
from .task import Task
from .warehouse import WarehouseMap


@dataclass
class WorldState:
    """Complete world snapshot returned after each simulation tick."""

    warehouse: WarehouseMap
    agents: list[Agent] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    tick: int = 0
    time: float = 0.0
    completed_task_ids_last_tick: list[int] = field(default_factory=list)
    collisions_last_tick: int = 0
    stalled_agent_ticks_last_tick: int = 0

    def agent_order(self) -> list[int]:
        return [agent.id for agent in sorted(self.agents, key=lambda item: item.id)]

    @property
    def step(self) -> int:
        return self.tick

    def get_task(self, task_id: int | None) -> Task | None:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def all_ports(self) -> list[Port]:
        return self.warehouse.loading_ports + self.warehouse.unloading_ports

    def get_port(self, port_id: int | None) -> Port | None:
        for port in self.all_ports():
            if port.id == port_id:
                return port
        return None

    def port_point(self, port_id: int | None) -> Point | None:
        port = self.get_port(port_id)
        if port is None:
            return None
        return port.operation_zone or port.entry_point or port.rect.center
