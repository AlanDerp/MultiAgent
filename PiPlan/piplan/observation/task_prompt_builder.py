from __future__ import annotations


class TaskPromptBuilder:
    DEFAULT_RULES = [
        "Plan coordinated collision-free warehouse motion for all agents.",
        "Use short smooth velocity actions in world coordinates.",
        "Keep clearance from walls, ports, obstacles, and other agents.",
        "Resolve congestion by assigning yield or escape motions to specific agents.",
        "Stop agents that have no active destination.",
    ]

    def build(self, global_state: dict, extra_rules: list[str] | None = None) -> str:
        parts = []
        for agent in sorted(global_state.get("agents", []), key=lambda item: item["id"]):
            destination = agent.get("destination")
            if destination is None:
                parts.append(f"agent {agent['id']} should stop")
                continue
            task = agent.get("task", {})
            load_state = "carrying item" if task.get("has_item") else "empty"
            parts.append(
                "agent {} is {}, task {}, target port {}, navigate to ({:.2f}, {:.2f})".format(
                    agent["id"],
                    load_state,
                    task.get("type"),
                    task.get("port_id"),
                    destination["x"],
                    destination["y"],
                )
            )
        rules = list(self.DEFAULT_RULES)
        if extra_rules:
            rules.extend(extra_rules)
        return "{} {} Rules: {}".format(self.DEFAULT_RULES[0], "; ".join(parts), " ".join(rules[1:]))
