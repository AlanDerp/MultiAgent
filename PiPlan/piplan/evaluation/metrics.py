from __future__ import annotations


"""Evaluation metrics for centralized multi-agent warehouse planning."""


def pph(world) -> float:
    """Picks per hour = completed tasks / simulated hours."""

    if world.time <= 0.0:
        return 0.0
    completed = sum(1 for task in world.tasks if task.status.value == "completed")
    return completed / (world.time / 3600.0)


def collision_rate(simulator) -> float:
    """Safety intervention count divided by total ticks."""

    ticks = max(simulator._state().step, 1)
    return len(getattr(simulator.actuator.safety, "safety_log", [])) / ticks


def stagnation_rate(simulator) -> float:
    """Stalled agent-ticks divided by all agent-ticks."""

    stalled = sum(item.get("stalled_agent_ticks", 0) for item in simulator.history)
    total = sum(item.get("agent_count", 0) for item in simulator.history)
    return stalled / total if total else 0.0


def average_task_completion_time(world) -> float:
    """Mean done_time - assign_time over completed tasks, in seconds."""

    durations = [
        task.done_time - task.assign_time
        for task in world.tasks
        if task.status.value == "completed" and task.done_time is not None and task.assign_time is not None
    ]
    return sum(durations) / len(durations) if durations else 0.0


def throughput_curve(simulator, window_ticks: int = 100) -> list[dict]:
    """Cumulative completed tasks sampled every `window_ticks` ticks."""

    curve = []
    for item in simulator.history:
        if item["tick"] % window_ticks == 0 or item["tick"] == simulator.history[-1]["tick"]:
            curve.append({"tick": item["tick"], "completed_total": item["completed_total"]})
    return curve


def average_distance_to_destination(world) -> float:
    distances = []
    for agent in world.agents:
        if agent.destination is not None:
            distances.append(agent.position.distance_to(agent.destination))
    return sum(distances) / len(distances) if distances else 0.0


def compute_metrics(simulator) -> dict:
    world = simulator._state()
    return {
        "pph": pph(world),
        "collision_rate": collision_rate(simulator),
        "stagnation_rate": stagnation_rate(simulator),
        "average_task_completion_time": average_task_completion_time(world),
        "throughput_curve": throughput_curve(simulator),
        "avg_distance_to_destination": average_distance_to_destination(world),
    }
