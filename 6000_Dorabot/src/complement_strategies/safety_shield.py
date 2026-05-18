from math import hypot
from .complement_strategy import ComplementStrategy


class SafetyShield(ComplementStrategy):
    """High-frequency safety layer for VLA actions."""

    def __init__(self, min_ray_distance=0.45, agent_clearance=0.8, wall_clearance=0.45, lookahead_time=0.35):
        self.min_ray_distance = min_ray_distance
        self.agent_clearance = agent_clearance
        self.wall_clearance = wall_clearance
        self.lookahead_time = lookahead_time

    def filter(self, agent, proposed_action, global_state):
        vx, vy = self._extract_velocity(proposed_action)
        vx, vy = self._clamp_speed(vx, vy, agent.cruise_speed)

        if self._yield_to_close_agent(agent, global_state):
            repaired = self._repair_velocity(agent, global_state)
            return {"vx": repaired[0], "vy": repaired[1], "reason": "agent_yield_repair"}

        if self._blocked_by_sensor(agent, vx, vy) or not self._is_safe_next_position(agent, global_state, vx, vy):
            repaired = self._repair_velocity(agent, global_state)
            if repaired != (0.0, 0.0):
                return {"vx": repaired[0], "vy": repaired[1], "reason": "repaired"}
            return {"vx": 0.0, "vy": 0.0, "reason": "stop_unsafe"}

        return {"vx": vx, "vy": vy, "reason": "pass"}

    def _extract_velocity(self, proposed_action):
        if hasattr(proposed_action, "vx") and hasattr(proposed_action, "vy"):
            return proposed_action.vx, proposed_action.vy
        if isinstance(proposed_action, dict):
            return proposed_action.get("vx", 0.0), proposed_action.get("vy", 0.0)
        return proposed_action[0], proposed_action[1]

    def _clamp_speed(self, vx, vy, max_speed):
        speed = hypot(vx, vy)
        if speed <= max_speed or speed == 0:
            return vx, vy
        ratio = max_speed / speed
        return vx * ratio, vy * ratio

    def _blocked_by_sensor(self, agent, vx, vy):
        if vx == 0.0 and vy == 0.0:
            return False
        if not agent.ray_length_list:
            return False
        return min(agent.ray_length_list) < self.min_ray_distance

    def _yield_to_close_agent(self, agent, global_state):
        for other in global_state["agents"]:
            if other["id"] == agent.id:
                continue
            dx = other["x"] - agent.position.x
            dy = other["y"] - agent.position.y
            if hypot(dx, dy) < self.agent_clearance and other["id"] < agent.id:
                return True
        return False

    def _repair_velocity(self, agent, global_state):
        vx, vy = self._goal_velocity(agent)
        vx, vy = self._add_static_repulsion(agent, global_state, vx, vy)
        vx, vy = self._clamp_speed(vx, vy, agent.cruise_speed * 0.6)
        if self._is_safe_next_position(agent, global_state, vx, vy):
            return (vx, vy)

        for candidate in self._escape_candidates(agent, global_state):
            if self._is_safe_next_position(agent, global_state, candidate[0], candidate[1]):
                return candidate
        return (0.0, 0.0)

    def _goal_velocity(self, agent):
        destination = agent.destination_location
        if destination is None:
            return (0.0, 0.0)
        dx = destination.x - agent.position.x
        dy = destination.y - agent.position.y
        distance = hypot(dx, dy)
        if distance == 0.0:
            return (0.0, 0.0)
        speed = agent.cruise_speed * 0.55
        return (dx / distance * speed, dy / distance * speed)

    def _add_static_repulsion(self, agent, global_state, vx, vy):
        map_info = global_state.get("map", {})
        width = float(map_info.get("width", 0.0))
        height = float(map_info.get("height", 0.0))
        x = agent.position.x
        y = agent.position.y
        margin = self._agent_radius(agent) + self.wall_clearance

        if width > 0.0:
            if x < margin:
                vx += agent.cruise_speed * 0.7
            elif x > width - margin:
                vx -= agent.cruise_speed * 0.7
        if height > 0.0:
            if y < margin:
                vy += agent.cruise_speed * 0.7
            elif y > height - margin:
                vy -= agent.cruise_speed * 0.7

        for obstacle in self._static_rectangles(global_state):
            cx = obstacle["x"] + obstacle["width"] * 0.5
            cy = obstacle["y"] + obstacle["height"] * 0.5
            dx = x - cx
            dy = y - cy
            distance = max(hypot(dx, dy), 1e-3)
            influence = max(obstacle["width"], obstacle["height"]) * 0.7 + margin
            if distance < influence:
                scale = (influence - distance) / influence * agent.cruise_speed
                vx += dx / distance * scale
                vy += dy / distance * scale
        return (vx, vy)

    def _escape_candidates(self, agent, global_state):
        speed = agent.cruise_speed * 0.45
        candidates = [
            (speed, 0.0),
            (-speed, 0.0),
            (0.0, speed),
            (0.0, -speed),
        ]
        map_info = global_state.get("map", {})
        width = float(map_info.get("width", 0.0))
        height = float(map_info.get("height", 0.0))
        x = agent.position.x
        y = agent.position.y
        margin = self._agent_radius(agent) + self.wall_clearance
        if width > 0.0 and x > width - margin:
            candidates.insert(0, (-speed, 0.0))
        if width > 0.0 and x < margin:
            candidates.insert(0, (speed, 0.0))
        if height > 0.0 and y > height - margin:
            candidates.insert(0, (0.0, -speed))
        if height > 0.0 and y < margin:
            candidates.insert(0, (0.0, speed))
        return candidates

    def _is_safe_next_position(self, agent, global_state, vx, vy):
        next_x = agent.position.x + vx * self.lookahead_time
        next_y = agent.position.y + vy * self.lookahead_time
        radius = self._agent_radius(agent)
        if not self._inside_map(global_state, next_x, next_y, radius):
            if self._reduces_map_violation(global_state, agent.position.x, agent.position.y, next_x, next_y, radius):
                return True
            return False
        for rect in self._static_rectangles(global_state):
            if self._inside_inflated_rect(next_x, next_y, rect, radius + self.wall_clearance):
                if self._reduces_rect_violation(agent.position.x, agent.position.y, next_x, next_y, rect):
                    continue
                return False
        return True

    def _inside_map(self, global_state, x, y, radius):
        map_info = global_state.get("map", {})
        width = float(map_info.get("width", 0.0))
        height = float(map_info.get("height", 0.0))
        if width <= 0.0 or height <= 0.0:
            return True
        margin = radius + self.wall_clearance
        return margin <= x <= width - margin and margin <= y <= height - margin

    def _reduces_map_violation(self, global_state, x, y, next_x, next_y, radius):
        map_info = global_state.get("map", {})
        width = float(map_info.get("width", 0.0))
        height = float(map_info.get("height", 0.0))
        if width <= 0.0 or height <= 0.0:
            return False
        margin = radius + self.wall_clearance

        def violation(px, py):
            return (
                max(0.0, margin - px)
                + max(0.0, px - (width - margin))
                + max(0.0, margin - py)
                + max(0.0, py - (height - margin))
            )

        current_violation = violation(x, y)
        next_violation = violation(next_x, next_y)
        return current_violation > 0.0 and next_violation < current_violation

    def _reduces_rect_violation(self, x, y, next_x, next_y, rect):
        cx = rect["x"] + rect["width"] * 0.5
        cy = rect["y"] + rect["height"] * 0.5
        return hypot(next_x - cx, next_y - cy) > hypot(x - cx, y - cy)

    def _static_rectangles(self, global_state):
        return (
            global_state.get("obstacles", [])
            + global_state.get("loading_ports", [])
            + global_state.get("unloading_ports", [])
        )

    def _inside_inflated_rect(self, x, y, rect, inflation):
        return (
            rect["x"] - inflation <= x <= rect["x"] + rect["width"] + inflation
            and rect["y"] - inflation <= y <= rect["y"] + rect["height"] + inflation
        )

    def _agent_radius(self, agent):
        try:
            return agent.shape.get_radius()
        except Exception:
            return 0.35
