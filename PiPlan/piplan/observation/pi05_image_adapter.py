from __future__ import annotations

import numpy as np


class Pi05ImageAdapter:
    """Render global warehouse state into top-down RGB.

    Input: render_state dict with map/agents/ports/obstacles in meters.
    Output: float32 array [3, size, size], normalized to [-1, 1].
    """

    def __init__(self, size: int = 224):
        self.size = size

    def to_joint_image(self, global_state: dict) -> np.ndarray:
        image = np.ones((3, self.size, self.size), dtype="float32")
        image[:] = 0.96
        width, height = self._map_size(global_state)

        for obstacle in global_state.get("obstacles", []):
            self._rect(image, obstacle, width, height, (0.18, 0.18, 0.18))
        for port in global_state.get("loading_ports", []):
            self._rect(image, port, width, height, (0.10, 0.45, 0.95))
        for port in global_state.get("unloading_ports", []):
            self._rect(image, port, width, height, (0.15, 0.70, 0.35))
        for agent in global_state.get("agents", []):
            destination = agent.get("destination")
            if destination:
                self._circle(image, destination["x"], destination["y"], width, height, 3, (0.90, 0.10, 0.75))
        for agent in global_state.get("agents", []):
            self._circle(image, agent["x"], agent["y"], width, height, 3, (0.95, 0.72, 0.10))
        return image * 2.0 - 1.0

    def _map_size(self, global_state: dict) -> tuple[float, float]:
        map_info = global_state.get("map", {})
        return max(float(map_info.get("width", 1.0)), 1.0), max(float(map_info.get("height", 1.0)), 1.0)

    def _rect(self, image, item, width, height, color):
        x0, y0 = self._pixel(item["x"], item["y"], width, height)
        x1, y1 = self._pixel(item["x"] + item["width"], item["y"] + item["height"], width, height)
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        image[:, max(0, top): min(self.size, bottom + 1), max(0, left): min(self.size, right + 1)] = np.array(
            color,
            dtype="float32",
        ).reshape(3, 1, 1)

    def _circle(self, image, x, y, width, height, radius, color):
        px, py = self._pixel(x, y, width, height)
        c = np.array(color, dtype="float32")
        for yy in range(max(0, py - radius), min(self.size, py + radius + 1)):
            for xx in range(max(0, px - radius), min(self.size, px + radius + 1)):
                if (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2:
                    image[:, yy, xx] = c

    def _pixel(self, x, y, width, height):
        px = int(max(0.0, min(1.0, x / width)) * (self.size - 1))
        py = int(max(0.0, min(1.0, y / height)) * (self.size - 1))
        return px, py
