from __future__ import annotations

from piplan.runtime.loop import run_demo
from .rollout import summarize_rollout


def smoke_benchmark(steps: int = 20) -> dict:
    return summarize_rollout(run_demo(steps=steps))
