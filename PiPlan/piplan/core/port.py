from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .geometry import Point, Rect


class PortType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class Port:
    """Warehouse port state.

    A port has a type (`input` or `output`) and a finite capacity measured in
    concurrent assigned agents/items.
    """

    id: int
    kind: str | PortType
    rect: Rect
    capacity: int = 1
    entry_point: Point | None = None
    operation_zone: Point | None = None
    queue_slots: list[Point] = field(default_factory=list)
    queue_agent_ids: list[int] = field(default_factory=list)
    num_items: int = 0
    assigned_agent_ids: list[int] = field(default_factory=list)

    @property
    def port_type(self) -> PortType:
        return self.kind if isinstance(self.kind, PortType) else PortType(str(self.kind))

    def can_accept(self) -> bool:
        return len(self.assigned_agent_ids) < self.capacity

    def assign(self, agent_id: int) -> bool:
        if agent_id in self.assigned_agent_ids:
            return True
        if not self.can_accept():
            return False
        self.assigned_agent_ids.append(agent_id)
        return True

    def release(self, agent_id: int) -> None:
        if agent_id in self.assigned_agent_ids:
            self.assigned_agent_ids.remove(agent_id)
