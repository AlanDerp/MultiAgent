from __future__ import annotations

from math import hypot

from piplan.core.action import VelocityAction
from piplan.core.agent import Agent
from piplan.core.geometry import Point
from piplan.core.state import WorldState


class SafetySupervisor:
    """Constraint layer for pi0.5 actions, not a planner.

    Input: proposed velocities in m/s.
    Output: constrained velocities in m/s and intervention events in
    `safety_log` / `last_tick_events`.
    """

    def __init__(
        self,
        max_speed: float = 1.5,
        wall_clearance: float = 0.15,
        safety_radius: float = 0.75,
        lookahead_time: float = 0.35,
    ):
        self.max_speed = max_speed
        self.wall_clearance = wall_clearance
        self.safety_radius = safety_radius
        self.lookahead_time = lookahead_time
        self.safety_log: list[dict] = []
        self.last_tick_events: list[dict] = []

    def filter(self, agent: Agent, proposed: VelocityAction, world: WorldState) -> VelocityAction:
        return self.filter_all(world, {agent.id: proposed})[agent.id]

    def filter_all(self, world: WorldState, proposed: dict[int, VelocityAction]) -> dict[int, VelocityAction]:
        self.last_tick_events = []
        agents_by_id = {agent.id: agent for agent in world.agents}
        constrained: dict[int, VelocityAction] = {}
        for agent_id, action in proposed.items():
            agent = agents_by_id[agent_id]
            vx, vy = self._clamp_speed(action.vx, action.vy, min(agent.cruise_speed, self.max_speed))
            next_point = agent.position.moved(vx, vy, self.lookahead_time)
            if not self._inside_map(agent, next_point, world):
                self._record(world, agent.id, "boundary_stop")
                constrained[agent_id] = VelocityAction(0.0, 0.0, reason="boundary_stop")
            elif self._hits_static(agent, next_point, world):
                self._record(world, agent.id, "static_stop")
                constrained[agent_id] = VelocityAction(0.0, 0.0, reason="static_stop")
            else:
                constrained[agent_id] = VelocityAction(vx, vy, reason=action.reason or "pass")
        self._apply_pairwise_emergency_stop(world, constrained)
        for agent in world.agents:
            constrained.setdefault(agent.id, VelocityAction(0.0, 0.0, reason="missing"))
        return constrained

    def _clamp_speed(self, vx: float, vy: float, max_speed: float) -> tuple[float, float]:
        speed = hypot(vx, vy)
        if speed <= max_speed or speed == 0.0:
            return vx, vy
        ratio = max_speed / speed
        return vx * ratio, vy * ratio

    def _apply_pairwise_emergency_stop(self, world: WorldState, actions: dict[int, VelocityAction]) -> None:
        agents = sorted(world.agents, key=lambda item: item.id)
        for idx, agent in enumerate(agents):
            for other in agents[idx + 1:]:
                if agent.position.distance_to(other.position) >= self.safety_radius:
                    continue
                first_speed = hypot(actions[agent.id].vx, actions[agent.id].vy)
                second_speed = hypot(actions[other.id].vx, actions[other.id].vy)
                stopped = agent if first_speed <= second_speed else other
                peer = other if stopped.id == agent.id else agent
                actions[stopped.id] = VelocityAction(0.0, 0.0, reason="emergency_agent_stop")
                self._record(world, stopped.id, "emergency_agent_stop", other_agent_id=peer.id)

    def _inside_map(self, agent: Agent, point: Point, world: WorldState) -> bool:
        margin = agent.radius + self.wall_clearance
        return margin <= point.x <= world.warehouse.width - margin and margin <= point.y <= world.warehouse.height - margin

    def _hits_static(self, agent: Agent, point: Point, world: WorldState) -> bool:
        inflation = agent.radius + self.wall_clearance
        for rect in world.warehouse.obstacles:
            if rect.inflated_contains(point, inflation):
                return True
        return False

    def _record(self, world: WorldState, agent_id: int, event: str, **extra) -> None:
        record = {"tick": world.step, "time": world.time, "agent_id": agent_id, "event": event, **extra}
        self.last_tick_events.append(record)
        self.safety_log.append(record)
