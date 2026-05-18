from __future__ import annotations

from piplan.core.action import JointActionChunk, VelocityAction
from piplan.core.geometry import Point, unit_towards
from piplan.observation.observation_bundle import Pi05ObservationBundle


class MockJointPolicy:
    """Deterministic pi0.5-shaped policy used to test the new runtime.

    Output: JointActionChunk with per-agent velocity actions in m/s.
    """

    def __init__(self, horizon_sec: float = 2.0, action_dt: float = 0.2, max_speed: float = 1.5):
        self.horizon_sec = horizon_sec
        self.action_dt = action_dt
        self.max_speed = max_speed

    def predict(self, observation: Pi05ObservationBundle) -> JointActionChunk:
        action_count = max(1, int(self.horizon_sec / self.action_dt))
        agents = sorted(observation.global_state.get("agents", []), key=lambda item: item["id"])
        frame = {}
        for agent in agents:
            destination = agent.get("destination")
            if not destination:
                frame[agent["id"]] = VelocityAction(0.0, 0.0, reason="no_destination")
                continue
            ux, uy = unit_towards(Point(agent["x"], agent["y"]), Point(destination["x"], destination["y"]))
            frame[agent["id"]] = VelocityAction(ux * self.max_speed, uy * self.max_speed, reason="mock_goal_direction")
        return JointActionChunk(
            agent_order=observation.agent_order,
            dt=self.action_dt,
            actions=[dict(frame) for _ in range(action_count)],
        )
