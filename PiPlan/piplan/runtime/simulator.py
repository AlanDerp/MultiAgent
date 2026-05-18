from __future__ import annotations

from piplan.control.actuator import Actuator
from piplan.observation.observation_bundle import Pi05ObservationBuilder
from piplan.policy.horizon_buffer import HorizonBuffer


class PiPlanSimulator:
    """pi0.5-first runtime loop.

    Data path per tick:
      WorldState -> Pi05ObservationBundle -> JointActionChunk -> safety
      constraints -> WarehouseWorld.step() -> WorldState.
    """

    def __init__(
        self,
        world,
        policy,
        dt: float = 0.2,
        observation_builder: Pi05ObservationBuilder | None = None,
        actuator: Actuator | None = None,
    ):
        self.world = world
        self.policy = policy
        self.dt = dt
        self.observation_builder = observation_builder or Pi05ObservationBuilder()
        self.actuator = actuator or Actuator()
        threshold = int(getattr(world, "config", {}).get("control", {}).get("replan_threshold", 2))
        self.horizon_buffer = HorizonBuffer(replan_threshold=threshold)
        self.last_observation = None
        self.last_action_log = None
        self.history: list[dict] = []

    def step(self):
        world_state = self._state()
        if self.horizon_buffer.should_replan():
            self.last_observation = self.observation_builder.build(world_state)
            chunk = self.policy.predict(self.last_observation)
            self.horizon_buffer.reset(chunk)
        agent_order = world_state.agent_order()
        proposed = self.horizon_buffer.pop(agent_order)
        self.last_action_log = self.actuator.apply(self._state(), proposed)
        for agent_id, action in self.last_action_log.applied.items():
            self.world.apply_velocity(agent_id, action.vx, action.vy)
        state = self.world.step(self.dt)
        self._record_history(state)
        return state

    def run(self, steps: int):
        last_state = self._state()
        for _ in range(steps):
            last_state = self.step()
        return last_state

    def _state(self):
        return getattr(self.world, "state", self.world)

    def _record_history(self, state) -> None:
        completed_total = sum(1 for task in state.tasks if task.status.value == "completed")
        self.history.append({
            "tick": state.step,
            "time": state.time,
            "completed_total": completed_total,
            "completed_this_tick": len(state.completed_task_ids_last_tick),
            "safety_events_this_tick": len(getattr(self.actuator.safety, "last_tick_events", [])),
            "stalled_agent_ticks": state.stalled_agent_ticks_last_tick,
            "agent_count": len(state.agents),
        })
