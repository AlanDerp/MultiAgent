from __future__ import annotations

from piplan.config import load_config
from piplan.core.agent import Agent
from piplan.core.geometry import Point, Rect
from piplan.core.port import Port, PortType
from piplan.core.state import WorldState
from piplan.core.task import Task, TaskManager, TaskStatus, TaskType
from piplan.core.warehouse import WarehouseMap
from piplan.runtime.physics import Box2DPhysics


class WarehouseWorld:
    """Integrated world runtime.

    Input: a WorldState plus config.
    step(dt): advances Box2D physics, task lifecycle, port capacity accounting,
    and returns the complete WorldState snapshot.
    """

    def __init__(self, state: WorldState, config: dict):
        self.state = state
        self.config = config
        self.task_manager = TaskManager(arrival_radius=float(config.get("tasks", {}).get("arrival_radius", 0.45)))
        self.physics = Box2DPhysics(state.warehouse, state.agents, config.get("physics", {}))
        self.task_manager.step(self.state)

    @property
    def agents(self):
        return self.state.agents

    @property
    def warehouse(self):
        return self.state.warehouse

    @property
    def tasks(self):
        return self.state.tasks

    @property
    def tick(self) -> int:
        return self.state.tick

    @property
    def time(self) -> float:
        return self.state.time

    @property
    def step_count(self) -> int:
        return self.state.tick

    def agent_order(self) -> list[int]:
        return self.state.agent_order()

    def apply_velocity(self, agent_id: int, vx: float, vy: float) -> None:
        self.physics.apply_velocity(agent_id, vx, vy)

    def step(self, dt: float) -> WorldState:
        self.physics.tick(dt)
        self._update_proximity_sensors()
        self.state.collisions_last_tick = self.physics.last_collision_count
        self.state.completed_task_ids_last_tick = self.task_manager.step(self.state)
        self.state.stalled_agent_ticks_last_tick = sum(
            1 for agent in self.state.agents
            if (agent.velocity.vx ** 2 + agent.velocity.vy ** 2) ** 0.5 < 0.05
        )
        self.state.tick += 1
        self.state.time += dt
        return self.state

    def pph(self) -> float:
        return self.task_manager.pph(self.state)

    def _update_proximity_sensors(self) -> None:
        max_range = 4.0
        for agent in self.state.agents:
            min_dist = max_range
            for other in self.state.agents:
                if other.id == agent.id:
                    continue
                min_dist = min(min_dist, max(0.0, agent.position.distance_to(other.position) - agent.radius - other.radius))
            wall_dist = min(
                agent.position.x,
                self.state.warehouse.width - agent.position.x,
                agent.position.y,
                self.state.warehouse.height - agent.position.y,
            )
            min_dist = min(min_dist, max(0.0, wall_dist - agent.radius))
            norm = min(1.0, min_dist / max_range)
            agent.ray_min = min_dist
            agent.ray_min_front = norm
            agent.ray_min_left = norm
            agent.ray_min_right = norm


def make_demo_world(config: dict | None = None, config_path: str | None = None, agent_count: int | None = None) -> WarehouseWorld:
    """Create a configured demo warehouse world for smoke rollouts."""

    cfg = config or load_config(config_path)
    map_cfg = cfg.get("map", {})
    agent_cfg = cfg.get("agents", {})
    width = float(map_cfg.get("width", 20.0))
    height = float(map_cfg.get("height", 20.0))
    count = int(agent_count if agent_count is not None else agent_cfg.get("count", 4))
    cruise_speed = float(agent_cfg.get("cruise_speed", cfg.get("policy", {}).get("max_speed", 1.5)))
    radius = float(agent_cfg.get("radius", 0.35))
    warehouse = _build_warehouse(cfg, width, height)
    agents = _build_agents(count, width, height, cruise_speed, radius)
    tasks = _build_tasks(cfg, warehouse, count)
    state = WorldState(warehouse=warehouse, agents=agents, tasks=tasks)
    return WarehouseWorld(state, cfg)


def _build_warehouse(cfg: dict, width: float, height: float) -> WarehouseMap:
    obstacles = [Rect(*[float(value) for value in item]) for item in cfg.get("map", {}).get("obstacles", [])]
    port_cfg = cfg.get("ports", {})
    input_count = int(port_cfg.get("input", {}).get("count", 2))
    output_count = int(port_cfg.get("output", {}).get("count", 2))
    input_capacity = int(port_cfg.get("input_capacity", 1))
    output_capacity = int(port_cfg.get("output_capacity", 1))
    loading_ports = []
    unloading_ports = []
    for idx in range(input_count):
        x = 1.0 + idx * max(2.0, width / max(input_count + 1, 2))
        rect = Rect(min(x, width - 2.0), 0.5, 1.0, 1.0)
        loading_ports.append(Port(id=idx, kind=PortType.INPUT, rect=rect, capacity=input_capacity, operation_zone=rect.center))
    for idx in range(output_count):
        x = 1.0 + idx * max(2.0, width / max(output_count + 1, 2))
        rect = Rect(min(x, width - 2.0), height - 1.5, 1.0, 1.0)
        unloading_ports.append(Port(id=100 + idx, kind=PortType.OUTPUT, rect=rect, capacity=output_capacity, operation_zone=rect.center))
    return WarehouseMap(width=width, height=height, obstacles=obstacles, loading_ports=loading_ports, unloading_ports=unloading_ports)


def _build_agents(count: int, width: float, height: float, cruise_speed: float, radius: float) -> list[Agent]:
    starts = [
        Point(2.0, 2.0),
        Point(width - 2.0, 2.0),
        Point(2.0, height - 2.0),
        Point(width - 2.0, height - 2.0),
    ]
    agents = []
    for idx in range(count):
        start = starts[idx % len(starts)]
        agents.append(Agent(id=idx, position=Point(start.x, start.y), cruise_speed=cruise_speed, radius=radius, state="idle"))
    return agents


def _build_tasks(cfg: dict, warehouse: WarehouseMap, agent_count: int) -> list[Task]:
    initial_count = int(cfg.get("tasks", {}).get("initial_count", max(agent_count, 1)))
    tasks = []
    if not warehouse.loading_ports or not warehouse.unloading_ports:
        return tasks
    for idx in range(initial_count):
        source = warehouse.loading_ports[idx % len(warehouse.loading_ports)]
        destination = warehouse.unloading_ports[idx % len(warehouse.unloading_ports)]
        tasks.append(
            Task(
                id=idx,
                task_type=TaskType.GO_TO_LOADING_PORT,
                status=TaskStatus.PENDING,
                source=source.operation_zone or source.rect.center,
                destination=source.operation_zone or source.rect.center,
                source_port_id=source.id,
                destination_port_id=destination.id,
                port_id=source.id,
                port_kind="input",
            )
        )
    return tasks
