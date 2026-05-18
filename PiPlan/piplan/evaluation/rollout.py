from __future__ import annotations

from piplan.config import load_config
from piplan.control.actuator import Actuator
from piplan.control.safety_supervisor import SafetySupervisor
from piplan.policy.action_decoder import JointActionDecoder
from piplan.policy.mock_policy import MockJointPolicy
from piplan.policy.pi05_client import Pi05PolicyClient
from piplan.policy.pi05_planner import Pi05JointPlanner
from piplan.runtime.simulator import PiPlanSimulator
from piplan.runtime.world import make_demo_world
from piplan.visualization.pygame_viewer import save_rollout_video
from .metrics import compute_metrics


def run_rollout(
    mode: str = "mock",
    steps: int = 100,
    agents: int = 4,
    config_path: str | None = None,
    model_id: str = "lerobot/pi05_base",
    device: str | None = None,
    lerobot_path: str = "../lerobot/src",
    save_video: str | None = None,
) -> tuple[PiPlanSimulator, dict]:
    """Run one rollout and return simulator plus metrics."""

    config = load_config(config_path)
    world = make_demo_world(config=config, agent_count=agents)
    policy_cfg = config.get("policy", {})
    runtime_cfg = config.get("runtime", {})
    control_cfg = config.get("control", {})
    max_speed = float(policy_cfg.get("max_speed", control_cfg.get("max_speed", 1.5)))
    if mode == "pi05":
        speeds = {agent.id: agent.cruise_speed for agent in world.agents}
        policy = Pi05JointPlanner(
            client=Pi05PolicyClient(model_id=model_id, device=device, lerobot_path=lerobot_path),
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
    frames = []
    for _ in range(steps):
        simulator.step()
        if save_video:
            frames.append(simulator.observation_builder.build(simulator._state()).map_image)
    if save_video:
        save_rollout_video(frames, save_video)
    metrics = compute_metrics(simulator)
    print_metrics(metrics)
    return simulator, metrics


def summarize_rollout(simulator) -> dict:
    return compute_metrics(simulator)


def print_metrics(metrics: dict) -> None:
    print(
        "metrics: pph={pph:.2f} collision_rate={collision_rate:.4f} stagnation_rate={stagnation_rate:.4f} "
        "avg_completion_time={average_task_completion_time:.2f}".format(**metrics)
    )
