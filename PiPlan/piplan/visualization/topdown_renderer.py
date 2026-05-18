from __future__ import annotations

from piplan.observation.global_state_encoder import GlobalStateEncoder


def render_topdown(world, size: int = 224):
    return GlobalStateEncoder(image_size=size).encode(world)["map_image"]
