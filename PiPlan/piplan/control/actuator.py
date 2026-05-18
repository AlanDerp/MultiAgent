from __future__ import annotations

from piplan.core.action import AppliedActionLog, VelocityAction
from piplan.core.state import WorldState
from .safety_supervisor import SafetySupervisor


class Actuator:
    """Applies proposed joint actions through SafetySupervisor.

    Input: WorldState and per-agent VelocityAction map in m/s.
    Output: AppliedActionLog with proposed and post-constraint actions.
    """

    def __init__(self, safety: SafetySupervisor | None = None):
        self.safety = safety or SafetySupervisor()

    def apply(self, world: WorldState, proposed: dict[int, VelocityAction]) -> AppliedActionLog:
        log = AppliedActionLog(proposed=dict(proposed), applied=self.safety.filter_all(world, proposed))
        for agent in world.agents:
            agent.velocity = log.applied.get(agent.id, VelocityAction(0.0, 0.0, reason="missing"))
        return log
