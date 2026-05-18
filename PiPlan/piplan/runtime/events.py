from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeEvent:
    step: int
    kind: str
    payload: dict
