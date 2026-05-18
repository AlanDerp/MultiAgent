from piplan.evaluation.rollout import summarize_rollout
from piplan.runtime.loop import run_demo


def test_runtime_smoke_moves_world_forward():
    simulator = run_demo(steps=5, agent_count=2)
    summary = summarize_rollout(simulator)

    assert simulator.world.step_count == 5
    assert simulator.world.time > 0.0
    assert summary["avg_distance_to_destination"] > 0.0
