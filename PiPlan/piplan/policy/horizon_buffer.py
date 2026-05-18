from __future__ import annotations

from collections import deque

from piplan.core.action import JointActionChunk, VelocityAction


class HorizonBuffer:
    """Stores and serves one joint action chunk.

    Input chunk shape: horizon x agent_id -> VelocityAction.
    Output per tick: agent_id -> VelocityAction for the current timestep.
    """

    def __init__(self, replan_threshold: int = 2):
        self._frames = deque()
        self.replan_threshold = replan_threshold

    def reset(self, chunk: JointActionChunk) -> None:
        self._frames = deque(chunk.actions)

    def empty(self) -> bool:
        return not self._frames

    def should_replan(self) -> bool:
        return len(self._frames) < self.replan_threshold

    def clear(self) -> None:
        self._frames.clear()

    def pop(self, agent_order: list[int]) -> dict[int, VelocityAction]:
        if not self._frames:
            return {agent_id: VelocityAction(0.0, 0.0, reason="buffer_empty") for agent_id in agent_order}
        frame = self._frames.popleft()
        return {
            agent_id: frame.get(agent_id, VelocityAction(0.0, 0.0, reason="missing_agent_action"))
            for agent_id in agent_order
        }
