from __future__ import annotations

import math
import sys
from pathlib import Path

from piplan.core.action import VelocityAction
from piplan.core.geometry import Point


"""Physics adapter for PiPlan.

Primary backend: pybox2d/Box2D with circular dynamic agent bodies and static
warehouse walls. Fallback backend: deterministic kinematics with the same
public interface, used only when Box2D cannot be imported in the active env.

Public output snapshot units:
  agent_id -> {"x": meters, "y": meters, "vx": m/s, "vy": m/s, "angle": radians}
"""


def _load_box2d():
    try:
        from Box2D import b2CircleShape, b2ContactListener, b2PolygonShape, b2World

        return b2World, b2CircleShape, b2PolygonShape, b2ContactListener
    except Exception:
        repo_root = Path(__file__).resolve().parents[3]
        build_candidates = [
            repo_root / "6000_Dorabot" / "pybox2d" / "build" / "lib.macosx-12.1-arm64-cpython-313",
            repo_root / "6000_Dorabot" / "pybox2d" / "build" / "lib.macosx-12.1-arm64-cpython-312",
            repo_root / "6000_Dorabot" / "pybox2d" / "build" / "lib.macosx-11.0-arm64-cpython-39",
        ]
        library_candidate = repo_root / "6000_Dorabot" / "pybox2d" / "library"
        for key in list(sys.modules):
            if key == "Box2D" or key.startswith("Box2D."):
                sys.modules.pop(key, None)
        for candidate in reversed(build_candidates):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        if library_candidate.exists() and str(library_candidate) not in sys.path:
            sys.path.insert(1, str(library_candidate))
        try:
            from Box2D import b2CircleShape, b2ContactListener, b2PolygonShape, b2World

            return b2World, b2CircleShape, b2PolygonShape, b2ContactListener
        except Exception:
            return None, None, None, None


class _CollisionListener:
    def __init__(self, base_cls):
        class Listener(base_cls):
            def __init__(self):
                super().__init__()
                self.collision_count = 0

            def BeginContact(self, contact):
                a = getattr(contact.fixtureA.body, "userData", None)
                b = getattr(contact.fixtureB.body, "userData", None)
                if isinstance(a, int) and isinstance(b, int):
                    self.collision_count += 1

        self.cls = Listener


class Box2DPhysics:
    """Box2D world wrapper.

    Inputs: WarehouseMap and Agent objects.
    Interface:
      - apply_velocity(agent_id, vx, vy): set dynamic body velocity in m/s.
      - tick(dt): advance physics and update Agent objects.
      - snapshot(): return all agent physical states.
    """

    def __init__(self, warehouse, agents, config: dict | None = None):
        self.warehouse = warehouse
        self.agents = {agent.id: agent for agent in agents}
        self.config = config or {}
        self.velocity_iterations = int(self.config.get("velocity_iterations", 8))
        self.position_iterations = int(self.config.get("position_iterations", 3))
        self.wall_thickness = float(self.config.get("wall_thickness", 0.1))
        self.last_collision_count = 0
        self.backend = "kinematic"
        self._world = None
        self._bodies = {}
        self._listener = None
        self._init_backend()

    def apply_velocity(self, agent_id: int, vx: float, vy: float) -> None:
        if self.backend == "box2d":
            body = self._bodies[agent_id]
            body.linearVelocity = (vx, vy)
            if vx != 0.0 or vy != 0.0:
                body.angle = math.atan2(vy, vx)
            return
        self.agents[agent_id].velocity = VelocityAction(vx, vy)

    def tick(self, dt: float) -> dict[int, dict]:
        if self.backend == "box2d":
            before = self._listener.collision_count if self._listener is not None else 0
            self._world.Step(dt, self.velocity_iterations, self.position_iterations)
            after = self._listener.collision_count if self._listener is not None else before
            self.last_collision_count = max(0, after - before)
            self._sync_agents_from_bodies()
        else:
            self.last_collision_count = 0
            for agent in self.agents.values():
                agent.step(dt)
                if agent.velocity.vx != 0.0 or agent.velocity.vy != 0.0:
                    agent.angle = math.atan2(agent.velocity.vy, agent.velocity.vx)
        return self.snapshot()

    def snapshot(self) -> dict[int, dict]:
        return {
            agent_id: {
                "x": agent.position.x,
                "y": agent.position.y,
                "vx": agent.velocity.vx,
                "vy": agent.velocity.vy,
                "angle": agent.angle,
            }
            for agent_id, agent in sorted(self.agents.items())
        }

    def _init_backend(self) -> None:
        b2World, b2CircleShape, b2PolygonShape, b2ContactListener = _load_box2d()
        if b2World is None:
            return
        listener_factory = _CollisionListener(b2ContactListener)
        self._listener = listener_factory.cls()
        self._world = b2World(gravity=(0, 0), doSleep=True, contactListener=self._listener)
        self._create_walls(b2PolygonShape)
        self._create_static_rectangles(b2PolygonShape)
        self._create_agents(b2CircleShape)
        self.backend = "box2d"

    def _create_walls(self, b2PolygonShape) -> None:
        width = self.warehouse.width
        height = self.warehouse.height
        t = self.wall_thickness
        wall_specs = [
            (width * 0.5, -t * 0.5, width * 0.5, t * 0.5),
            (width * 0.5, height + t * 0.5, width * 0.5, t * 0.5),
            (-t * 0.5, height * 0.5, t * 0.5, height * 0.5),
            (width + t * 0.5, height * 0.5, t * 0.5, height * 0.5),
        ]
        for cx, cy, hx, hy in wall_specs:
            body = self._world.CreateStaticBody(position=(cx, cy))
            body.CreatePolygonFixture(box=(hx, hy), density=0.0)

    def _create_static_rectangles(self, b2PolygonShape) -> None:
        for rect in self.warehouse.obstacles:
            body = self._world.CreateStaticBody(position=(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5))
            body.CreatePolygonFixture(box=(rect.width * 0.5, rect.height * 0.5), density=0.0)
        for port in self.warehouse.loading_ports + self.warehouse.unloading_ports:
            body = self._world.CreateStaticBody(position=(port.rect.x + port.rect.width * 0.5, port.rect.y + port.rect.height * 0.5))
            body.CreatePolygonFixture(box=(port.rect.width * 0.5, port.rect.height * 0.5), density=0.0, isSensor=True)

    def _create_agents(self, b2CircleShape) -> None:
        for agent in self.agents.values():
            body = self._world.CreateDynamicBody(position=(agent.position.x, agent.position.y), angle=agent.angle, userData=agent.id)
            body.CreateCircleFixture(radius=agent.radius, density=1.0, friction=0.1, restitution=0.0)
            self._bodies[agent.id] = body

    def _sync_agents_from_bodies(self) -> None:
        for agent_id, body in self._bodies.items():
            agent = self.agents[agent_id]
            agent.position = Point(float(body.position.x), float(body.position.y))
            agent.velocity = VelocityAction(float(body.linearVelocity.x), float(body.linearVelocity.y))
            agent.angle = float(body.angle)
