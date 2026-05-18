from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return hypot(self.x - other.x, self.y - other.y)

    def moved(self, vx: float, vy: float, dt: float) -> "Point":
        return Point(self.x + vx * dt, self.y + vy * dt)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point:
        return Point(self.x + self.width * 0.5, self.y + self.height * 0.5)

    def inflated_contains(self, point: Point, inflation: float) -> bool:
        return (
            self.x - inflation <= point.x <= self.x + self.width + inflation
            and self.y - inflation <= point.y <= self.y + self.height + inflation
        )


def unit_towards(source: Point, target: Point) -> tuple[float, float]:
    dx = target.x - source.x
    dy = target.y - source.y
    distance = hypot(dx, dy)
    if distance == 0.0:
        return 0.0, 0.0
    return dx / distance, dy / distance
