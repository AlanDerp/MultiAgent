from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import Rect
from .port import Port


@dataclass
class WarehouseMap:
    width: float
    height: float
    obstacles: list[Rect] = field(default_factory=list)
    loading_ports: list[Port] = field(default_factory=list)
    unloading_ports: list[Port] = field(default_factory=list)
