from __future__ import annotations

from math import tanh

from piplan.core.action import JointActionChunk, VelocityAction


class JointActionDecoder:
    """Decode raw pi0.5 actions into a JointActionChunk.

    Input: raw_actions shape [horizon, 2*N].
    Output: per-timestep per-agent vx/vy in m/s, aligned to agent_order.
    """

    def __init__(
        self,
        action_dt: float = 0.2,
        speed_by_agent: dict[int, float] | None = None,
        default_speed: float = 1.5,
        max_speed: float | None = None,
        output_mode: str = "raw_velocity",
    ):
        self.action_dt = action_dt
        self.speed_by_agent = speed_by_agent or {}
        self.default_speed = max_speed if max_speed is not None else default_speed
        if output_mode not in {"raw_velocity", "normalized_tanh"}:
            raise ValueError(f"Unsupported action output mode: {output_mode}")
        self.output_mode = output_mode

    def from_raw(self, raw_chunk, agent_order: list[int]) -> JointActionChunk:
        rows = self._normalize(raw_chunk)
        frames = []
        for row in rows:
            frame = {}
            for idx, agent_id in enumerate(agent_order):
                vx_idx = idx * 2
                vy_idx = vx_idx + 1
                raw_vx = float(row[vx_idx]) if len(row) > vx_idx else 0.0
                raw_vy = float(row[vy_idx]) if len(row) > vy_idx else 0.0
                speed = self.speed_by_agent.get(agent_id, self.default_speed)
                if self.output_mode == "normalized_tanh":
                    vx = tanh(raw_vx) * speed
                    vy = tanh(raw_vy) * speed
                else:
                    vx = self._clamp(raw_vx, -speed, speed)
                    vy = self._clamp(raw_vy, -speed, speed)
                frame[agent_id] = VelocityAction(vx, vy, reason=f"pi05:{self.output_mode}")
            frames.append(frame)
        if not frames:
            frames.append({agent_id: VelocityAction(0.0, 0.0, reason="empty_chunk") for agent_id in agent_order})
        return JointActionChunk(agent_order=agent_order, dt=self.action_dt, actions=frames, raw=raw_chunk)

    def _normalize(self, raw_chunk) -> list[list[float]]:
        if hasattr(raw_chunk, "tolist"):
            raw_chunk = raw_chunk.tolist()
        rows = raw_chunk
        while rows and isinstance(rows, list) and isinstance(rows[0], list) and len(rows) == 1:
            if rows[0] and isinstance(rows[0][0], list):
                rows = rows[0]
            else:
                break
        if rows and not isinstance(rows[0], list):
            rows = [rows]
        return rows or [[]]

    def _clamp(self, value: float, low: float, high: float) -> float:
        return min(max(value, low), high)
