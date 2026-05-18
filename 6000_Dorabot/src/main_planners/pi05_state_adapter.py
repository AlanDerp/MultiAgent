import math


class Pi05StateAdapter:
    """Flatten simulator state V into the OBS_STATE vector pi05 expects."""

    def __init__(self, max_agents=8):
        self.max_agents = max_agents

    def to_vector(self, ego_agent, global_state):
        destination = ego_agent.destination_location
        vector = [
            ego_agent.position.x,
            ego_agent.position.y,
            ego_agent.linear_velocity[0],
            ego_agent.linear_velocity[1],
            math.cos(ego_agent.angle),
            math.sin(ego_agent.angle),
        ]
        if destination is None:
            vector.extend([0.0, 0.0, 0.0])
        else:
            dx = destination.x - ego_agent.position.x
            dy = destination.y - ego_agent.position.y
            vector.extend([dx, dy, math.hypot(dx, dy)])

        others = [agent for agent in global_state["agents"] if agent["id"] != ego_agent.id]
        others.sort(key=lambda agent: (agent["x"] - ego_agent.position.x) ** 2 + (agent["y"] - ego_agent.position.y) ** 2)
        for other in others[: self.max_agents - 1]:
            vector.extend([
                other["x"] - ego_agent.position.x,
                other["y"] - ego_agent.position.y,
                other["vx"],
                other["vy"],
            ])

        ray_lengths = ego_agent.ray_length_list or []
        if ray_lengths:
            vector.extend([
                min(ray_lengths),
                sum(ray_lengths) / len(ray_lengths),
                ray_lengths[0],
                ray_lengths[len(ray_lengths) // 2],
                ray_lengths[-1],
            ])
        else:
            vector.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        return vector


class JointPi05StateAdapter:
    """Flatten the full multi-agent state into one joint OBS_STATE vector."""

    def __init__(self, max_agents=16):
        self.max_agents = max_agents

    def to_vector(self, global_state):
        vector = []
        agents = sorted(global_state.get("agents", []), key=lambda agent: agent["id"])
        map_info = global_state.get("map", {})
        vector.extend([
            float(map_info.get("width", 0.0)),
            float(map_info.get("height", 0.0)),
            float(len(agents)),
        ])

        for agent in agents[: self.max_agents]:
            destination = agent.get("destination")
            task = agent.get("task", {})
            if destination is None:
                dx = 0.0
                dy = 0.0
                distance = 0.0
                has_destination = 0.0
            else:
                dx = destination["x"] - agent["x"]
                dy = destination["y"] - agent["y"]
                distance = math.hypot(dx, dy)
                has_destination = 1.0
            vector.extend([
                float(agent["id"]),
                float(agent["x"]),
                float(agent["y"]),
                float(agent["vx"]),
                float(agent["vy"]),
                float(agent["angle"]),
                float(dx),
                float(dy),
                float(distance),
                float(has_destination),
                float(agent.get("ray_min") or 0.0),
                self._task_code(task.get("type")),
                self._port_kind_code(task.get("port_kind")),
                1.0 if task.get("has_item") else 0.0,
                float(task.get("port_id") if task.get("port_id") is not None else -1.0),
                float(task.get("item_destination_port_id") if task.get("item_destination_port_id") is not None else -1.0),
            ])
        return vector

    def _task_code(self, task_type):
        if task_type is None:
            return 0.0
        if "GO_TO_LOADING_PORT" in task_type:
            return 1.0
        if "GO_TO_UNLOADING_PORT" in task_type:
            return 2.0
        if "WAITING_FOR_ORDER_TASK" in task_type:
            return 0.0
        return -1.0

    def _port_kind_code(self, port_kind):
        if port_kind == "LoadingPort":
            return 1.0
        if port_kind == "UnloadingPort":
            return 2.0
        return 0.0
