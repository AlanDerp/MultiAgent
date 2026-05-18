from __future__ import annotations

from piplan.observation.observation_bundle import Pi05ObservationBundle
from .action_decoder import JointActionDecoder
from .pi05_client import Pi05PolicyClient


class Pi05JointPlanner:
    def __init__(self, client: Pi05PolicyClient | None = None, decoder: JointActionDecoder | None = None):
        self.client = client or Pi05PolicyClient()
        self.decoder = decoder or JointActionDecoder()
        self.last_raw_chunk = None

    def predict(self, observation: Pi05ObservationBundle):
        self.last_raw_chunk = self.client.predict_action_chunk(
            observation.state_vector,
            observation.task,
            image=observation.image,
        )
        return self.decoder.from_raw(self.last_raw_chunk, observation.agent_order)
