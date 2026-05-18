class Pi05ImageAdapter:
    """Render global warehouse state into a simple top-down RGB observation."""

    def __init__(self, size=224):
        self.size = size

    def to_image(self, ego_agent, global_state):
        import numpy as np

        image = np.ones((3, self.size, self.size), dtype="float32")
        image[:] = 0.96
        map_info = global_state.get("map", {})
        width = max(float(map_info.get("width", 1.0)), 1.0)
        height = max(float(map_info.get("height", 1.0)), 1.0)

        for obstacle in global_state.get("obstacles", []):
            self._rect(image, obstacle, width, height, (0.18, 0.18, 0.18))
        for port in global_state.get("loading_ports", []):
            self._rect(image, port, width, height, (0.10, 0.45, 0.95))
        for port in global_state.get("unloading_ports", []):
            self._rect(image, port, width, height, (0.15, 0.70, 0.35))

        destination = ego_agent.destination_location
        if destination is not None:
            self._circle(image, destination.x, destination.y, width, height, 4, (0.90, 0.10, 0.75))

        for agent in global_state.get("agents", []):
            color = (0.90, 0.15, 0.12) if agent["id"] == ego_agent.id else (0.95, 0.72, 0.10)
            self._circle(image, agent["x"], agent["y"], width, height, 3, color)
        return image

    def to_joint_image(self, global_state):
        import numpy as np

        image = np.ones((3, self.size, self.size), dtype="float32")
        image[:] = 0.96
        map_info = global_state.get("map", {})
        width = max(float(map_info.get("width", 1.0)), 1.0)
        height = max(float(map_info.get("height", 1.0)), 1.0)

        for obstacle in global_state.get("obstacles", []):
            self._rect(image, obstacle, width, height, (0.18, 0.18, 0.18))
        for port in global_state.get("loading_ports", []):
            self._rect(image, port, width, height, (0.10, 0.45, 0.95))
        for port in global_state.get("unloading_ports", []):
            self._rect(image, port, width, height, (0.15, 0.70, 0.35))

        for agent in global_state.get("agents", []):
            destination = agent.get("destination")
            if destination is not None:
                self._circle(image, destination["x"], destination["y"], width, height, 3, (0.90, 0.10, 0.75))
        for agent in global_state.get("agents", []):
            self._circle(image, agent["x"], agent["y"], width, height, 3, (0.95, 0.72, 0.10))
        return image

    def _rect(self, image, item, width, height, color):
        x0, y0 = self._pixel(item["x"], item["y"], width, height)
        x1, y1 = self._pixel(item["x"] + item["width"], item["y"] + item["height"], width, height)
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        image[:, max(0, top): min(self.size, bottom + 1), max(0, left): min(self.size, right + 1)] = self._color_block(color)

    def _circle(self, image, x, y, width, height, radius, color):
        px, py = self._pixel(x, y, width, height)
        c = self._color_pixel(color)
        for yy in range(max(0, py - radius), min(self.size, py + radius + 1)):
            for xx in range(max(0, px - radius), min(self.size, px + radius + 1)):
                if (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2:
                    image[:, yy, xx] = c

    def _pixel(self, x, y, width, height):
        px = int(max(0.0, min(1.0, x / width)) * (self.size - 1))
        py = int(max(0.0, min(1.0, y / height)) * (self.size - 1))
        return px, py

    def _color_block(self, color):
        import numpy as np

        return np.array(color, dtype="float32").reshape(3, 1, 1)

    def _color_pixel(self, color):
        import numpy as np

        return np.array(color, dtype="float32")
