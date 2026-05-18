import os
import sys


class Pi05PolicyClient:
    """Thin LeRobot pi05 adapter.

    It mirrors the working pattern in the sibling lerobot/testpi05.py script:
    load PI05Policy, create the official pre/post processors, then call
    predict_action_chunk for horizon actions.
    """

    def __init__(self, model_id="lerobot/pi05_base", device=None, lerobot_path=None, seed=42):
        self.model_id = model_id
        self.device_name = device
        self.lerobot_path = lerobot_path or self._default_lerobot_path()
        self.seed = seed
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.torch = None
        self.ACTION = None
        self.OBS_STATE = None
        self.image_keys = []
        self.state_dim = None
        self.action_dim = None

    def predict_action_chunk(self, state_vector, task, image=None):
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
        if self.lerobot_path and self.lerobot_path not in sys.path:
            sys.path.insert(0, self.lerobot_path)

        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            from lerobot.utils.constants import ACTION, OBS_STATE
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pi05 dependencies are not available. In the LeRobot repo, install them with "
                "`pip install -e \".[pi]\"` inside the active conda environment."
            ) from exc

        self.torch = torch
        self.ACTION = ACTION
        self.OBS_STATE = OBS_STATE
        torch.manual_seed(self.seed)
        device_name = self.device_name or self._auto_device(torch)
        self.device_name = device_name

        self.policy = PI05Policy.from_pretrained(self.model_id, strict=True)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.model_id,
            preprocessor_overrides={"device_processor": {"device": str(device_name)}},
        )
        self.image_keys = list(self.policy.config.image_features.keys())
        self.state_dim = (
            self.policy.config.input_features.get(OBS_STATE).shape[0]
            if self.policy.config.input_features and OBS_STATE in self.policy.config.input_features
            else self.policy.config.max_state_dim
        )
        self.action_dim = (
            self.policy.config.output_features.get(ACTION).shape[0]
            if self.policy.config.output_features and ACTION in self.policy.config.output_features
            else self.policy.config.max_action_dim
        )

    def _tensor(self, values, target_dim):
        padded = list(values[:target_dim])
        if len(padded) < target_dim:
            padded.extend([0.0] * (target_dim - len(padded)))
        return self.torch.tensor(padded, dtype=self.torch.float32)

    def _image_tensor(self, image):
        if image is not None:
            if hasattr(image, "detach"):
                return image.to(dtype=self.torch.float32)
            return self.torch.tensor(image, dtype=self.torch.float32)
        return self.torch.zeros(3, 224, 224, dtype=self.torch.float32)

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

    def _default_lerobot_path(self):
        current = os.path.abspath(os.path.dirname(__file__))
        repo_root = os.path.abspath(os.path.join(current, "..", ".."))
        candidate = os.path.abspath(os.path.join(repo_root, "..", "lerobot", "src"))
        if os.path.isdir(candidate):
            return candidate
        candidate = os.path.abspath(os.path.join(repo_root, "..", "lerobot"))
        if os.path.isdir(candidate):
            return candidate
        return None
