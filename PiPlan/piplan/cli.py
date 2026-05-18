from __future__ import annotations

import argparse

from piplan.policy.mock_policy import MockJointPolicy
from piplan.runtime.simulator import PiPlanSimulator
from piplan.runtime.world import make_demo_world


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PiPlan pi0.5-first warehouse planning runtime.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-demo", help="Run the mock pi0.5-shaped runtime smoke demo.")
    run_parser.add_argument("--steps", type=int, default=20)
    run_parser.add_argument("--agents", type=int, default=4)
    run_parser.add_argument("--dt", type=float, default=0.2)

    args = parser.parse_args(argv)
    if args.command == "run-demo":
        world = make_demo_world(agent_count=args.agents)
        simulator = PiPlanSimulator(world=world, policy=MockJointPolicy(action_dt=args.dt), dt=args.dt)
        simulator.run(args.steps)
        for agent in simulator.world.agents:
            print(
                "agent {} pos=({:.2f},{:.2f}) vel=({:.2f},{:.2f})".format(
                    agent.id,
                    agent.position.x,
                    agent.position.y,
                    agent.velocity.vx,
                    agent.velocity.vy,
                )
            )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
