from __future__ import annotations

PIPLAN_ROLLOUT_SCHEMA = "piplan_rollout_v1"
PIPLAN_EXPERT_SCHEMA = "piplan_expert_v1"


def pad(values, dim: int) -> list[float]:
    result = [float(value) for value in values[:dim]]
    if len(result) < dim:
        result.extend([0.0] * (dim - len(result)))
    return result
