from __future__ import annotations

import numpy as np


class JointPi05StateAdapter:
    """Flatten per-agent tokens for LeRobot `observation.state`.

    Input: encoder output with `map_features` shape [3] and
    `agent_tokens` shape [N, 16].
    Output: flat Python list length 3 + N * 16, ordered by `agent_order`.
    Metadata: `agent_order` is copied separately by ObservationBuilder.
    """

    def __init__(self, pad_dim: int = 0):
        self.pad_dim = pad_dim

    def to_vector(self, encoded: dict | np.ndarray | list) -> list[float]:
        if isinstance(encoded, dict):
            map_features = np.asarray(encoded.get("map_features", []), dtype="float32").reshape(-1)
            tokens = np.asarray(encoded.get("agent_tokens", []), dtype="float32").reshape(-1)
            vector = np.concatenate([map_features, tokens]).astype("float32").tolist()
        else:
            array = np.asarray(encoded, dtype="float32")
            vector = array.reshape(-1).astype("float32").tolist()
        if self.pad_dim:
            return self._pad(vector, self.pad_dim)
        return vector

    def _pad(self, values: list[float], dim: int) -> list[float]:
        padded = values[:dim]
        if len(padded) < dim:
            padded.extend([0.0] * (dim - len(padded)))
        return padded
