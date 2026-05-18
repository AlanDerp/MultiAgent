from collections import deque

from geometry import Point, compute_direction
from complement_strategies import SafetyShield
from .action_schema import TimedAction
from .pi05_image_adapter import Pi05ImageAdapter
from .pi05_policy_client import Pi05PolicyClient
from .pi05_state_adapter import JointPi05StateAdapter
from .prompt_builder import PromptBuilder
from .state_encoder import GlobalStateEncoder


class CentralizedVLACoordinator:
    """One VLA policy that plans joint actions for all agents."""

    def __init__(self, use_pi05=False, model_id="lerobot/pi05_base", device=None, lerobot_path=None,
                 horizon_sec=2.0, action_dt=0.2):
        self.use_pi05 = use_pi05
        self.horizon_sec = horizon_sec
        self.action_dt = action_dt
        self.encoder = GlobalStateEncoder()
        self.state_adapter = JointPi05StateAdapter()
        self.image_adapter = Pi05ImageAdapter()
        self.prompt_builder = PromptBuilder()
        self.complement_strategy = SafetyShield()
        self.action_buffers = {}
        self.last_global_state = None
        self.last_joint_state_vector = None
        self.last_task_text = None
        self.last_raw_chunk = None
        self.client = None
        if use_pi05:
            self.client = Pi05PolicyClient(model_id=model_id, device=device, lerobot_path=lerobot_path)

    def reset_agent(self, agent):
        self.action_buffers.pop(agent.id, None)

    def plan_and_apply(self, agents, server):
        global_state = self.encoder.encode_all(server)
        self.last_global_state = global_state
        self.last_joint_state_vector = self.state_adapter.to_vector(global_state)
        self.last_task_text = self._task_text(global_state)

        if self._should_replan(agents):
            joint_plan = self._plan_joint_actions(agents, global_state)
            self.action_buffers = {agent_id: deque(actions) for agent_id, actions in joint_plan.items()}
            for agent in agents:
                agent.goal_changed = False
                agent.replan = False

        for agent in agents:
            agent.last_bc_global_state = global_state
            agent.last_bc_state_vector = self.last_joint_state_vector
            agent.last_joint_task_text = self.last_task_text
            agent.last_joint_global_state = global_state
            if not agent.has_destination():
                agent.stop()
                continue

            buffer = self.action_buffers.get(agent.id, deque())
            proposed = buffer.popleft() if buffer else TimedAction(self.action_dt, 0.0, 0.0)
            self.action_buffers[agent.id] = buffer
            safe = self.complement_strategy.filter(agent, proposed, global_state)
            agent.last_proposed_action = proposed
            agent.last_safe_action = safe
            agent.last_shield_reason = safe.get("reason")
            agent.linear_velocity = (safe["vx"], safe["vy"])

    def _should_replan(self, agents):
        if not self.action_buffers:
            return True
        if any(agent.goal_changed or agent.replan for agent in agents):
            return True
        return any(not self.action_buffers.get(agent.id) for agent in agents if agent.has_destination())

    def _plan_joint_actions(self, agents, global_state):
        if self.use_pi05:
            return self._plan_with_pi05(agents, global_state)
        return self._plan_with_mock_policy(agents)

    def _plan_with_mock_policy(self, agents):
        action_count = max(1, int(self.horizon_sec / self.action_dt))
        plans = {}
        for agent in agents:
            if not agent.has_destination():
                plans[agent.id] = [TimedAction(self.action_dt, 0.0, 0.0) for _ in range(action_count)]
                continue
            direction = compute_direction(agent.position, agent.destination_location)
            action = TimedAction(
                self.action_dt,
                direction.x * agent.cruise_speed,
                direction.y * agent.cruise_speed,
            )
            plans[agent.id] = [action for _ in range(action_count)]
        return plans

    def _plan_with_pi05(self, agents, global_state):
        image = self.image_adapter.to_joint_image(global_state)
        raw_chunk = self.client.predict_action_chunk(self.last_joint_state_vector, self.last_task_text, image=image)
        self.last_raw_chunk = raw_chunk
        rows = self._normalize_chunk(raw_chunk)
        action_count = max(1, int(self.horizon_sec / self.action_dt))
        plans = {agent.id: [] for agent in agents}
        ordered_agents = sorted(agents, key=lambda agent: agent.id)
        for row in rows[:action_count]:
            for idx, agent in enumerate(ordered_agents):
                vx_idx = idx * 2
                vy_idx = vx_idx + 1
                vx = float(row[vx_idx]) if len(row) > vx_idx else 0.0
                vy = float(row[vy_idx]) if len(row) > vy_idx else 0.0
                plans[agent.id].append(TimedAction(
                    self.action_dt,
                    self._clamp(vx, -agent.cruise_speed, agent.cruise_speed),
                    self._clamp(vy, -agent.cruise_speed, agent.cruise_speed),
                ))
        for agent in agents:
            if not plans[agent.id]:
                plans[agent.id].append(TimedAction(self.action_dt, 0.0, 0.0))
        return plans

    def _normalize_chunk(self, raw_chunk):
        rows = raw_chunk
        while rows and isinstance(rows[0], list) and len(rows) == 1 and rows[0] and isinstance(rows[0][0], list):
            rows = rows[0]
        if rows and not isinstance(rows[0], list):
            rows = [rows]
        return rows or [[]]

    def _clamp(self, value, low, high):
        return min(max(value, low), high)

    def _task_text(self, global_state):
        active = []
        for agent in sorted(global_state.get("agents", []), key=lambda item: item["id"]):
            destination = agent.get("destination")
            if destination is None:
                active.append(f"agent {agent['id']} should stop")
            else:
                task = agent.get("task", {})
                task_type = task.get("type")
                has_item = "carrying item" if task.get("has_item") else "empty"
                port_kind = task.get("port_kind") or "unknown port"
                active.append(
                    "agent {} is {}, task {}, target {} {}, navigate to assigned waypoint ({:.2f}, {:.2f})".format(
                        agent["id"],
                        has_item,
                        task_type,
                        port_kind,
                        task.get("port_id"),
                        destination["x"],
                        destination["y"],
                    )
                )
        rules = " ".join(PromptBuilder.DEFAULT_RULES + [
            "Loading and unloading ports are not final geometric goals; agents must follow assigned entry, queue, and operation waypoints.",
            "After loading, carry the item to its assigned unloading port. After unloading, request a new loading task.",
            "Resolve multi-agent deadlocks by assigning clear yield or escape motions to specific agents.",
        ])
        return "Plan coordinated collision-free warehouse motion for all agents. {}. Rules: {}".format(
            "; ".join(active), rules
        )
