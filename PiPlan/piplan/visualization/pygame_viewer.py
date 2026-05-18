from __future__ import annotations

from pathlib import Path

import numpy as np


def save_rollout_video(frames: list[np.ndarray], path: str | Path) -> Path:
    """Best-effort video writer.

    Input frames are [3,H,W] float arrays normalized to [-1,1].
    If imageio is unavailable, saves a compressed `.npz` at the requested path
    plus `.npz` suffix. This keeps rollout evaluation functional without GUI
    dependencies.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    images = [((frame + 1.0) * 127.5).clip(0, 255).astype("uint8").transpose(1, 2, 0) for frame in frames]
    try:
        import imageio.v2 as imageio

        imageio.mimsave(target, images, fps=10)
        return target
    except Exception:
        fallback = target.with_suffix(target.suffix + ".npz")
        np.savez_compressed(fallback, frames=np.asarray(images, dtype="uint8"))
        return fallback
