import json
import os
import time

class DataLogger:
    def __init__(self, frequency=10, filename=None, mode="basic", include_global_state=False):
        self.frequency = frequency
        self.mode = mode
        self.include_global_state = include_global_state
        if filename:
            self.filename = filename
        else:
            suffix = "joint_bc" if mode == "joint_bc" else ("bc" if mode == "bc" else "log")
            self.filename = f"data_samples/{suffix}_{int(time.time())}.jsonl"
            
        os.makedirs(os.path.dirname(os.path.abspath(self.filename)), exist_ok=True)
        self.records = []

    def log_step(self, simulator_time, task_count, agents, simulator=None):
        if self.mode == "joint_bc":
            self._log_joint_bc_step(simulator_time, task_count, agents, simulator)
            return
        if self.mode == "bc":
            self._log_bc_step(simulator_time, task_count, agents, simulator)
            return
        self._log_basic_step(simulator_time, task_count, agents)

    def _log_basic_step(self, simulator_time, task_count, agents):
        import math
        step_data = {
            "schema_version": "basic_v1",
            "time": simulator_time,
            "task_count": task_count,
            "agents": []
        }
        for agent in agents:
            real_speed = math.hypot(agent.linear_velocity[0], agent.linear_velocity[1])
            agent_data = {
                "id": agent.id,
                "x": agent.position.x,
                "y": agent.position.y,
                "angle": agent.angle,
                "speed": real_speed,
                "destination_x": agent.destination_location.x if agent.destination_location else None,
                "destination_y": agent.destination_location.y if agent.destination_location else None
            }
            step_data["agents"].append(agent_data)
        
        self.records.append(step_data)
        self._flush_if_needed()

    def _log_bc_step(self, simulator_time, task_count, agents, simulator):
        from main_planners.state_encoder import GlobalStateEncoder
        from main_planners.pi05_state_adapter import Pi05StateAdapter

        if simulator is None:
            return
        encoder = GlobalStateEncoder()
        adapter = Pi05StateAdapter()
        for agent in agents:
            if not agent.has_destination():
                continue
            global_state = getattr(agent, "last_bc_global_state", None) or encoder.encode(agent, agent.server)
            state_vector = getattr(agent, "last_bc_state_vector", None) or adapter.to_vector(agent, global_state)
            record = {
                "schema_version": "bc_v1",
                "time": simulator_time,
                "step": simulator.step_counter,
                "task_count": task_count,
                "agent_id": agent.id,
                "planner": {
                    "agent_class": type(agent).__name__,
                    "global_planner": type(agent.global_planner).__name__ if agent.global_planner else None,
                    "local_planner": type(agent.current_local_planner).__name__ if agent.current_local_planner else None,
                    "main_planner": type(agent.main_planner).__name__ if hasattr(agent, "main_planner") else None,
                    "complement_strategy": type(agent.complement_strategy).__name__ if hasattr(agent, "complement_strategy") else None,
                },
                "observation": {
                    "state_vector": state_vector,
                    "ray_min": min(agent.ray_length_list) if agent.ray_length_list else None,
                    "ray_sample": self._sample_rays(agent.ray_length_list),
                    "ego": self._agent_state(agent),
                    "nearby_agents": self._nearby_agents(agent, global_state),
                    "map": global_state.get("map", {}),
                },
                "language": {
                    "rules": getattr(agent, "language_rules", []),
                    "task_text": self._task_text(agent),
                },
                "action": {
                    "executed_velocity": self._tuple(agent.linear_velocity),
                    "expert_velocity": self._tuple(getattr(agent, "last_expert_action", agent.linear_velocity)),
                    "proposed_velocity": self._tuple(getattr(agent, "last_proposed_action", None)),
                    "safe_velocity": self._tuple(getattr(agent, "last_safe_action", None)),
                    "shield_reason": getattr(agent, "last_shield_reason", None),
                },
                "target": {
                    "destination": self._point(agent.destination_location),
                    "distance_to_destination": agent.position.distance(agent.destination_location) if agent.destination_location else None,
                },
                "task": self._task(agent),
            }
            if self.include_global_state:
                record["global_state"] = global_state
            self.records.append(record)
        self._flush_if_needed()

    def _log_joint_bc_step(self, simulator_time, task_count, agents, simulator):
        from main_planners.state_encoder import GlobalStateEncoder
        from main_planners.pi05_state_adapter import JointPi05StateAdapter

        if simulator is None:
            return
        coordinator = getattr(simulator, "vla_coordinator", None)
        if coordinator is not None and coordinator.last_global_state is not None:
            global_state = coordinator.last_global_state
            state_vector = coordinator.last_joint_state_vector
            task_text = coordinator.last_task_text
        else:
            global_state = GlobalStateEncoder().encode_all(simulator.server)
            state_vector = JointPi05StateAdapter().to_vector(global_state)
            task_text = self._joint_task_text(global_state)

        agent_actions = []
        joint_expert_action = []
        joint_executed_action = []
        joint_proposed_action = []
        joint_safe_action = []
        for agent in sorted(agents, key=lambda item: item.id):
            executed = self._tuple(agent.linear_velocity)
            expert = self._tuple(getattr(agent, "last_expert_action", agent.linear_velocity))
            proposed = self._tuple(getattr(agent, "last_proposed_action", None))
            safe = self._tuple(getattr(agent, "last_safe_action", None))
            agent_actions.append({
                "agent_id": agent.id,
                "executed_velocity": executed,
                "expert_velocity": expert,
                "proposed_velocity": proposed,
                "safe_velocity": safe,
                "shield_reason": getattr(agent, "last_shield_reason", None),
                "destination": self._point(agent.destination_location),
                "task": self._task(agent),
            })
            joint_executed_action.extend(executed or [0.0, 0.0])
            joint_expert_action.extend(expert or executed or [0.0, 0.0])
            joint_proposed_action.extend(proposed or [0.0, 0.0])
            joint_safe_action.extend(safe or executed or [0.0, 0.0])

        record = {
            "schema_version": "joint_bc_v1",
            "time": simulator_time,
            "step": simulator.step_counter,
            "task_count": task_count,
            "planner": {
                "mode": "centralized_vla" if coordinator is not None else "centralized_expert",
                "main_planner": type(coordinator).__name__ if coordinator is not None else None,
            },
            "observation": {
                "joint_state_vector": state_vector,
                "map": global_state.get("map", {}),
                "agents": global_state.get("agents", []),
            },
            "language": {
                "task_text": task_text,
            },
            "action": {
                "joint_executed_action": joint_executed_action,
                "joint_expert_action": joint_expert_action,
                "joint_proposed_action": joint_proposed_action,
                "joint_safe_action": joint_safe_action,
                "agents": agent_actions,
            },
        }
        if self.include_global_state:
            record["global_state"] = global_state
        self.records.append(record)
        self._flush_if_needed()

    def _joint_task_text(self, global_state):
        parts = []
        for agent in sorted(global_state.get("agents", []), key=lambda item: item["id"]):
            destination = agent.get("destination")
            if destination is None:
                parts.append(f"agent {agent['id']} should stop")
            else:
                parts.append(
                    "agent {} navigate to ({:.2f}, {:.2f})".format(
                        agent["id"], destination["x"], destination["y"]
                    )
                )
        return "Plan coordinated collision-free warehouse motion for all agents. " + "; ".join(parts)

    def _flush_if_needed(self):
        if len(self.records) >= 1000:
            self.flush()

    def _sample_rays(self, rays, count=16):
        if not rays:
            return []
        if len(rays) <= count:
            return list(rays)
        result = []
        for idx in range(count):
            source_idx = int(idx * (len(rays) - 1) / (count - 1))
            result.append(rays[source_idx])
        return result

    def _nearby_agents(self, agent, global_state, limit=8):
        others = [other for other in global_state["agents"] if other["id"] != agent.id]
        others.sort(key=lambda other: (other["x"] - agent.position.x) ** 2 + (other["y"] - agent.position.y) ** 2)
        return others[:limit]

    def _agent_state(self, agent):
        return {
            "x": agent.position.x,
            "y": agent.position.y,
            "angle": agent.angle,
            "vx": agent.linear_velocity[0],
            "vy": agent.linear_velocity[1],
            "state": str(agent.state),
        }

    def _task_text(self, agent):
        destination = agent.destination_location
        if destination is None:
            return "stop because there is no destination"
        return "navigate to destination ({:.2f}, {:.2f})".format(destination.x, destination.y)

    def _task(self, agent):
        task = getattr(agent, "task", None)
        item = getattr(agent, "current_item", None)
        port = getattr(task, "port", None)
        return {
            "state": str(getattr(agent, "state", None)),
            "type": str(getattr(task, "type", None)),
            "port_id": getattr(port, "identifier", None),
            "port_kind": type(port).__name__ if port is not None else None,
            "has_item": item is not None,
            "item_source_port_id": getattr(item, "source_port_id", None),
            "item_destination_port_id": getattr(item, "destination_port_id", None),
            "last_event": getattr(agent, "last_task_event", None),
        }

    def _tuple(self, value):
        if value is None:
            return None
        if hasattr(value, "vx") and hasattr(value, "vy"):
            return [value.vx, value.vy]
        if isinstance(value, dict):
            return [value.get("vx", 0.0), value.get("vy", 0.0)]
        return [value[0], value[1]]

    def _point(self, point):
        if point is None:
            return None
        return {"x": point.x, "y": point.y}
        
    def flush(self):
        if not self.records:
            return
            
        with open(self.filename, 'a') as f:
            for record in self.records:
                f.write(json.dumps(record) + "\n")
        self.records = []

    def export_data(self):
        self.flush()
        print(f"\n[DataLogger] Data export completed. Saved to {self.filename}")
