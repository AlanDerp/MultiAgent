class GlobalStateEncoder:
    """Build a compact V state for VLA-style planning."""

    def encode_all(self, server):
        agents = []
        for agent in server.agents:
            destination = agent.destination_location
            agents.append({
                "id": agent.id,
                "is_ego": False,
                "x": agent.position.x,
                "y": agent.position.y,
                "vx": agent.linear_velocity[0],
                "vy": agent.linear_velocity[1],
                "angle": agent.angle,
                "state": str(agent.state),
                "task": self._agent_task(agent),
                "destination": None if destination is None else {
                    "x": destination.x,
                    "y": destination.y,
                },
                "ray_min": min(agent.ray_length_list) if agent.ray_length_list else None,
            })

        return {
            "ego_agent_id": None,
            "map": self._map(server),
            "agents": agents,
            "loading_ports": self._ports(server.loading_ports),
            "unloading_ports": self._ports(server.unloading_ports),
            "obstacles": self._obstacles(server),
        }

    def encode(self, ego_agent, server):
        agents = []
        for agent in server.agents:
            destination = agent.destination_location
            agents.append({
                "id": agent.id,
                "is_ego": agent.id == ego_agent.id,
                "x": agent.position.x,
                "y": agent.position.y,
                "vx": agent.linear_velocity[0],
                "vy": agent.linear_velocity[1],
                "angle": agent.angle,
                "state": str(agent.state),
                "task": self._agent_task(agent),
                "destination": None if destination is None else {
                    "x": destination.x,
                    "y": destination.y,
                },
                "ray_min": min(agent.ray_length_list) if agent.ray_length_list else None,
            })

        return {
            "ego_agent_id": ego_agent.id,
            "map": self._map(server),
            "agents": agents,
            "loading_ports": self._ports(server.loading_ports),
            "unloading_ports": self._ports(server.unloading_ports),
            "obstacles": self._obstacles(server),
        }

    def _map(self, server):
        environment = getattr(server, "environment", None)
        if environment is None:
            return {"width": 0.0, "height": 0.0}
        return {
            "width": environment.width_in_meters,
            "height": environment.height_in_meters,
        }

    def _ports(self, ports):
        result = []
        for port in ports:
            result.append({
                "id": port.identifier,
                "kind": type(port).__name__,
                "x": port.location.x,
                "y": port.location.y,
                "width": port.dimension[0],
                "height": port.dimension[1],
                "entry_point": self._point(port.entry_point),
                "operation_zone": self._point(port.operation_zone),
                "queue_slots": [self._point(slot) for slot in port.queue.slots] if port.queue else [],
                "queue_agent_ids": [agent.id for agent in port.queue.agents] if port.queue else [],
                "num_items": len(getattr(port, "items", [])),
            })
        return result

    def _obstacles(self, server):
        environment = getattr(server, "environment", None)
        if environment is None:
            return []
        result = []
        for obstacle in list(environment.obstacles.values()):
            result.append({
                "id": obstacle.identifier,
                "x": obstacle.location.x,
                "y": obstacle.location.y,
                "width": obstacle.dimension[0],
                "height": obstacle.dimension[1],
            })
        return result

    def _agent_task(self, agent):
        task = getattr(agent, "task", None)
        item = getattr(agent, "current_item", None)
        port = getattr(task, "port", None)
        return {
            "type": str(getattr(task, "type", None)),
            "port_id": getattr(port, "identifier", None),
            "port_kind": type(port).__name__ if port is not None else None,
            "source": self._point(getattr(task, "source_location", None)),
            "destination": self._point(getattr(task, "destination_location", None)),
            "has_item": item is not None,
            "item_source_port_id": getattr(item, "source_port_id", None),
            "item_destination_port_id": getattr(item, "destination_port_id", None),
            "last_event": getattr(agent, "last_task_event", None),
        }

    def _point(self, point):
        if point is None:
            return None
        return {"x": point.x, "y": point.y}
