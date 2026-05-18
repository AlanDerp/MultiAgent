from dataclasses import dataclass


@dataclass
class TimedAction:
    dt: float
    vx: float
    vy: float

    def to_velocity_tuple(self):
        return (self.vx, self.vy)


@dataclass
class HorizonPlan:
    agent_id: int
    horizon_sec: float
    actions: list
    waypoints: list
    intent: str = "cruise"
    risk: str = "unknown"

