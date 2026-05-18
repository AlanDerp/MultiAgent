#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piplan.config import load_config
from piplan.control.actuator import Actuator
from piplan.control.safety_supervisor import SafetySupervisor
from piplan.policy.action_decoder import JointActionDecoder
from piplan.policy.mock_policy import MockJointPolicy
from piplan.policy.pi05_client import Pi05PolicyClient
from piplan.policy.pi05_planner import Pi05JointPlanner
from piplan.evaluation.metrics import compute_metrics
from piplan.runtime.simulator import PiPlanSimulator
from piplan.runtime.world import make_demo_world


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "pi05"], default="mock")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--model_id", default="lerobot/pi05_base")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"])
    parser.add_argument("--lerobot_path", default="../lerobot/src")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    world = make_demo_world(config=config, agent_count=args.agents)
    policy_cfg = config.get("policy", {})
    runtime_cfg = config.get("runtime", {})
    control_cfg = config.get("control", {})
    max_speed = float(policy_cfg.get("max_speed", control_cfg.get("max_speed", 1.5)))
    if args.mode == "pi05":
        speeds = {agent.id: agent.cruise_speed for agent in world.agents}
        policy = Pi05JointPlanner(
            client=Pi05PolicyClient(model_id=args.model_id, device=args.device, lerobot_path=args.lerobot_path),
            decoder=JointActionDecoder(
                action_dt=float(policy_cfg.get("action_dt", runtime_cfg.get("dt", 0.2))),
                speed_by_agent=speeds,
                max_speed=max_speed,
                output_mode=str(policy_cfg.get("action_output_mode", "raw_velocity")),
            ),
        )
    else:
        policy = MockJointPolicy(
            horizon_sec=float(policy_cfg.get("horizon_sec", 2.0)),
            action_dt=float(policy_cfg.get("action_dt", runtime_cfg.get("dt", 0.2))),
            max_speed=max_speed,
        )
    actuator = Actuator(SafetySupervisor(
        max_speed=float(control_cfg.get("max_speed", max_speed)),
        safety_radius=float(control_cfg.get("safety_radius", 0.75)),
        wall_clearance=float(control_cfg.get("wall_clearance", 0.15)),
        lookahead_time=float(control_cfg.get("lookahead_time", 0.35)),
    ))
    simulator = PiPlanSimulator(world=world, policy=policy, dt=float(runtime_cfg.get("dt", 0.2)), actuator=actuator)
    final_state = simulator.run(args.steps)
    metrics = compute_metrics(simulator)
    print(
        "metrics: ticks={} pph={:.2f} collision_rate={:.4f} stagnation_rate={:.4f} completed={}".format(
            final_state.step,
            metrics["pph"],
            metrics["collision_rate"],
            metrics["stagnation_rate"],
            len([task for task in final_state.tasks if task.status.value == "completed"]),
        )
    )
    for agent in final_state.agents:
        print(f"agent {agent.id}: ({agent.position.x:.2f}, {agent.position.y:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
