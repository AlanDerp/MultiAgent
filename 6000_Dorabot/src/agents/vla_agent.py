from collections import deque
from agents.naive_agent import NaiveAgent
from agents.agent_state_machine import AgentState
from main_planners import MockPi05MainPlanner, Pi05MainPlanner
from main_planners.state_encoder import GlobalStateEncoder
from main_planners.pi05_state_adapter import Pi05StateAdapter
from complement_strategies import SafetyShield


class VLAAgent(NaiveAgent):
    """Agent controlled by a VLA-style main planner plus safety complement."""

    def __init__(self, *args, **kwargs):
        super(VLAAgent, self).__init__(*args, **kwargs)
        self.main_planner = MockPi05MainPlanner()
        self.complement_strategy = SafetyShield()
        self.global_state_encoder = GlobalStateEncoder()
        self.action_buffer = deque()
        self.language_rules = []
        self.last_horizon_plan = None
        self.last_global_state = None
        self.last_proposed_action = None
        self.last_safe_action = None
        self.last_shield_reason = None
        self.bc_state_adapter = Pi05StateAdapter()

    def use_pi05_main_planner(self, model_id="lerobot/pi05_base", device=None, lerobot_path=None):
        self.main_planner = Pi05MainPlanner(model_id=model_id, device=device, lerobot_path=lerobot_path)

    def prepare_for_centralized_plan(self):
        func = self.state_machine.next_state(self)
        func(self, self.server)

        if not self.has_destination():
            self.action_buffer.clear()
            self.stop()
            return False

        if self.state_machine.arrive_at_destination(self.position, self.destination_location):
            if self.state == AgentState.CRUISE:
                self.go_internal_stations()
            else:
                self.internal_stations = []
                self.action_buffer.clear()
                self.stop()
            return False

        if self.goal_changed:
            self.destination_location = self.task.destination_location
            self.action_buffer.clear()
        return True

    def plan(self):
        """Legacy decentralized VLA plan path.

        The simulator now uses CentralizedVLACoordinator in VLA mode. This is
        kept for isolated debugging only.
        """
        func = self.state_machine.next_state(self)
        func(self, self.server)

        if not self.has_destination():
            self.action_buffer.clear()
            self.stop()
            return

        if self.state_machine.arrive_at_destination(self.position, self.destination_location):
            if self.state == AgentState.CRUISE:
                self.go_internal_stations()
            else:
                self.internal_stations = []
                self.action_buffer.clear()
                self.stop()
            return

        if self.goal_changed:
            self.destination_location = self.task.destination_location
            self.action_buffer.clear()

        self.last_global_state = self.global_state_encoder.encode(self, self.server)
        self.last_bc_global_state = self.last_global_state
        self.last_bc_state_vector = self.bc_state_adapter.to_vector(self, self.last_global_state)
        if self.main_planner.should_replan(self, self.action_buffer):
            self.last_horizon_plan = self.main_planner.plan(self, self.last_global_state, self.language_rules)
            self.action_buffer = deque(self.last_horizon_plan.actions)
            self.goal_changed = False
            self.replan = False

        proposed_action = self.action_buffer.popleft() if self.action_buffer else {"vx": 0.0, "vy": 0.0}
        safe_action = self.complement_strategy.filter(self, proposed_action, self.last_global_state)
        self.last_proposed_action = proposed_action
        self.last_safe_action = safe_action
        self.last_shield_reason = safe_action.get("reason")
        self.linear_velocity = (safe_action["vx"], safe_action["vy"])
