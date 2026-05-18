from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from piplan.core.state import WorldState
from .global_state_encoder import GlobalStateEncoder
from .pi05_state_adapter import JointPi05StateAdapter


@dataclass
class Pi05ObservationBundle:
    """Centralized pi0.5 observation.

    agent_tokens: float32 [N, 16].
    map_image: float32 [3, 224, 224], normalized to [-1, 1].
    state_vector: flat LeRobot-compatible `observation.state`, length 3 + N * 16.
    action alignment: `agent_order[i]` maps to raw action columns [2*i, 2*i+1].
    """

    agent_tokens: np.ndarray
    map_image: np.ndarray
    task_text: str
    agent_order: list[int]
    state_vector: list[float]
    metadata: dict

    @property
    def image(self) -> np.ndarray:
        return self.map_image

    @property
    def task(self) -> str:
        return self.task_text

    @property
    def global_state(self) -> dict:
        return self.metadata.get("render_state", {})


class Pi05ObservationBuilder:
    def __init__(
        self,
        encoder: GlobalStateEncoder | None = None,
        state_adapter: JointPi05StateAdapter | None = None,
    ):
        self.encoder = encoder or GlobalStateEncoder()
        self.state_adapter = state_adapter or JointPi05StateAdapter()

    def build(self, world: WorldState) -> Pi05ObservationBundle:
        encoded = self.encoder.encode(world)
        return Pi05ObservationBundle(
            agent_tokens=encoded["agent_tokens"],
            map_image=encoded["map_image"],
            task_text=encoded["task_text"],
            agent_order=encoded["agent_order"],
            state_vector=self.state_adapter.to_vector(encoded),
            metadata=encoded["metadata"],
        )
