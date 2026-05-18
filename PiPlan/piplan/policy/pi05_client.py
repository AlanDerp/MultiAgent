from __future__ import annotations

import os
import sys


class Pi05PolicyClient:
    """Thin LeRobot pi0.5 adapter with the official pre/post processors."""

    def __init__(self, model_id: str = "lerobot/pi05_base", device: str | None = None, lerobot_path: str | None = None, seed: int = 42):
        self.model_id = model_id
        self.device_name = device
        self.lerobot_path = lerobot_path
        self.seed = seed
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.torch = None
        self.OBS_STATE = None
        self.image_keys = []
        self.state_dim = None

    def predict_action_chunk(self, state_vector, task: str, image=None):
        self._ensure_loaded()
        batch = {
            self.OBS_STATE: self._tensor(state_vector, self.state_dim),
            "task": task,
        }
        for key in self.image_keys:
            batch[key] = self._image_tensor(image)
        batch_proc = self.preprocessor(batch)
        with self.torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch_proc)
            chunk = self.postprocessor(chunk)
        return self._to_python(chunk)

    def _ensure_loaded(self):
        if self.policy is not None:
            return
        if self.lerobot_path:
            path = os.path.abspath(self.lerobot_path)
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            from lerobot.utils.constants import OBS_STATE
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pi0.5 dependencies are unavailable. Install LeRobot with `pip install -e \".[pi]\"`."
            ) from exc

        self.torch = torch
        self.OBS_STATE = OBS_STATE
        torch.manual_seed(self.seed)
        device_name = self.device_name or self._auto_device(torch)
        self.policy = PI05Policy.from_pretrained(self.model_id, strict=True)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.model_id,
            preprocessor_overrides={"device_processor": {"device": str(device_name)}},
        )
        self.image_keys = list(self.policy.config.image_features.keys())
        feature = self.policy.config.input_features.get(OBS_STATE) if self.policy.config.input_features else None
        self.state_dim = feature.shape[0] if feature else self.policy.config.max_state_dim

    def _tensor(self, values, target_dim):
        padded = list(values[:target_dim])
        if len(padded) < target_dim:
            padded.extend([0.0] * (target_dim - len(padded)))
        return self.torch.tensor(padded, dtype=self.torch.float32)

    def _image_tensor(self, image):
        if image is None:
            return self.torch.zeros(3, 224, 224, dtype=self.torch.float32)
        if hasattr(image, "detach"):
            return image.to(dtype=self.torch.float32)
        return self.torch.tensor(image, dtype=self.torch.float32)

    def _to_python(self, chunk):
        if hasattr(chunk, "detach"):
            chunk = chunk.detach().cpu()
        return chunk.tolist()

    def _auto_device(self, torch):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
