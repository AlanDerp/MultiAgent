class MainPlanner:
    def should_replan(self, agent, action_buffer):
        return not action_buffer or agent.goal_changed or agent.replan

    def plan(self, agent, global_state, language_rules=None):
        raise NotImplementedError

