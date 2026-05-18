from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReplayBuffer:
    frames: list[dict] = field(default_factory=list)

    def add(self, frame: dict) -> None:
        self.frames.append(frame)

    def clear(self) -> None:
        self.frames.clear()
