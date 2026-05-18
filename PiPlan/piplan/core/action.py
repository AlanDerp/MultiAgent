from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VelocityAction:
    vx: float
    vy: float
    reason: str | None = None

    def as_list(self) -> list[float]:
        return [self.vx, self.vy]


@dataclass(frozen=True)
class TimedAction(VelocityAction):
    dt: float = 0.2


@dataclass
class JointActionChunk:
    agent_order: list[int]
    dt: float
    actions: list[dict[int, VelocityAction]]
    raw: object | None = None

    @property
    def horizon_sec(self) -> float:
        return self.dt * len(self.actions)

    def is_empty(self) -> bool:
        return not self.actions


@dataclass
class AppliedActionLog:
    proposed: dict[int, VelocityAction] = field(default_factory=dict)
    applied: dict[int, VelocityAction] = field(default_factory=dict)
