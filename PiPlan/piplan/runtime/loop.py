from __future__ import annotations

from piplan.config import load_config
from piplan.policy.mock_policy import MockJointPolicy
from .simulator import PiPlanSimulator
from .world import make_demo_world


def run_demo(steps: int = 20, agent_count: int = 4, config_path: str | None = None) -> PiPlanSimulator:
    config = load_config(config_path)
    world = make_demo_world(config=config, agent_count=agent_count)
    policy_cfg = config.get("policy", {})
    runtime_cfg = config.get("runtime", {})
    simulator = PiPlanSimulator(
        world=world,
        policy=MockJointPolicy(
            horizon_sec=float(policy_cfg.get("horizon_sec", 2.0)),
            action_dt=float(policy_cfg.get("action_dt", runtime_cfg.get("dt", 0.2))),
            max_speed=float(policy_cfg.get("max_speed", 1.5)),
        ),
        dt=float(runtime_cfg.get("dt", 0.2)),
    )
    simulator.run(steps)
    return simulator
