from geometry import Point, compute_direction
from .action_schema import HorizonPlan, TimedAction
from .main_planner import MainPlanner
from .pi05_image_adapter import Pi05ImageAdapter
from .pi05_policy_client import Pi05PolicyClient
from .pi05_state_adapter import Pi05StateAdapter
from .prompt_builder import PromptBuilder


class MockPi05MainPlanner(MainPlanner):
    """pi05-compatible planner stub.

    This deliberately keeps the same input/output shape expected from a real
    VLA model while using a deterministic goal-directed policy for now.
    """

    def __init__(self, horizon_sec=2.0, action_dt=0.2):
        self.horizon_sec = horizon_sec
        self.action_dt = action_dt
        self.prompt_builder = PromptBuilder()
        self.last_prompt = None

    def plan(self, agent, global_state, language_rules=None):
        self.last_prompt = self.prompt_builder.build(global_state, language_rules)
        destination = agent.destination_location
        if destination is None:
            return HorizonPlan(agent.id, self.horizon_sec, [], [], intent="stop", risk="low")

        direction = compute_direction(agent.position, destination)
        speed = agent.cruise_speed
        action_count = max(1, int(self.horizon_sec / self.action_dt))
        action = TimedAction(self.action_dt, direction.x * speed, direction.y * speed)
        actions = [action for _ in range(action_count)]
        waypoints = self._rollout_waypoints(agent.position, action, action_count)
        return HorizonPlan(
            agent_id=agent.id,
            horizon_sec=self.horizon_sec,
            actions=actions,
            waypoints=waypoints,
            intent="go_to_destination",
            risk="low",
        )

    def _rollout_waypoints(self, start, action, action_count):
        points = []
        x = start.x
        y = start.y
        for _ in range(action_count):
            x += action.vx * action.dt
            y += action.vy * action.dt
            points.append(Point(x, y))
        return points


class Pi05MainPlanner(MainPlanner):
    """Real pi05-backed main planner with a mock-compatible output contract."""

    def __init__(self, model_id="lerobot/pi05_base", device=None, lerobot_path=None, horizon_sec=2.0, action_dt=0.2):
        self.horizon_sec = horizon_sec
        self.action_dt = action_dt
        self.prompt_builder = PromptBuilder()
        self.state_adapter = Pi05StateAdapter()
        self.image_adapter = Pi05ImageAdapter()
        self.client = Pi05PolicyClient(model_id=model_id, device=device, lerobot_path=lerobot_path)
        self.last_prompt = None

    def plan(self, agent, global_state, language_rules=None):
        self.last_prompt = self.prompt_builder.build(global_state, language_rules)
        state_vector = self.state_adapter.to_vector(agent, global_state)
        image = self.image_adapter.to_image(agent, global_state)
        task = self._task_text(agent, self.last_prompt["rules"])
        raw_chunk = self.client.predict_action_chunk(state_vector, task, image=image)
        actions = self._chunk_to_actions(raw_chunk, agent)
        waypoints = self._rollout_waypoints(agent.position, actions)
        return HorizonPlan(
            agent_id=agent.id,
            horizon_sec=sum(action.dt for action in actions),
            actions=actions,
            waypoints=waypoints,
            intent="pi05_go_to_destination",
            risk="model_generated",
        )

    def _task_text(self, agent, rules):
        destination = agent.destination_location
        if destination is None:
            goal = "stop because there is no destination"
        else:
            goal = "navigate to destination ({:.2f}, {:.2f})".format(destination.x, destination.y)
        return "{}. Rules: {}".format(goal, " ".join(rules))

    def _chunk_to_actions(self, raw_chunk, agent):
        rows = raw_chunk
        while rows and isinstance(rows[0], list) and rows and len(rows) == 1 and rows[0] and isinstance(rows[0][0], list):
            rows = rows[0]
        if rows and not isinstance(rows[0], list):
            rows = [rows]

        actions = []
        max_count = max(1, int(self.horizon_sec / self.action_dt))
        for row in rows[:max_count]:
            vx = float(row[0]) if len(row) > 0 else 0.0
            vy = float(row[1]) if len(row) > 1 else 0.0
            actions.append(TimedAction(
                self.action_dt,
                self._clamp(vx, -agent.cruise_speed, agent.cruise_speed),
                self._clamp(vy, -agent.cruise_speed, agent.cruise_speed),
            ))
        if not actions:
            actions.append(TimedAction(self.action_dt, 0.0, 0.0))
        return actions

    def _clamp(self, value, low, high):
        return min(max(value, low), high)

    def _rollout_waypoints(self, start, actions):
        points = []
        x = start.x
        y = start.y
        for action in actions:
            x += action.vx * action.dt
            y += action.vy * action.dt
            points.append(Point(x, y))
        return points
