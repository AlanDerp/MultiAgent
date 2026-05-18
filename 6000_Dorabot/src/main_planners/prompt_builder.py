class PromptBuilder:
    """Collect language-side planning rules L.

    The mock planner does not call a model yet, but keeping this boundary now
    makes the pi05 client swap small and explicit later.
    """

    DEFAULT_RULES = [
        "Complete the assigned warehouse task with short, smooth motion.",
        "Keep a safety margin around obstacles, ports, walls, and other agents.",
        "When a nearby agent has lower id, yield unless the ego agent is almost at its destination.",
        "If the local situation is unsafe or ambiguous, slow down or stop.",
        "Return a short horizon of velocity actions in world coordinates.",
    ]

    def build(self, global_state, extra_rules=None):
        rules = list(self.DEFAULT_RULES)
        if extra_rules:
            rules.extend(extra_rules)
        return {
            "rules": rules,
            "state": global_state,
            "output_contract": {
                "horizon_sec": "float",
                "actions": [{"dt": "float", "vx": "float", "vy": "float"}],
                "waypoints": [["x", "y"]],
                "intent": "string",
                "risk": "low|medium|high",
            },
        }

