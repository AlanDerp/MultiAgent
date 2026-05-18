from __future__ import annotations

from piplan.legacy_expert.dorabot_log_reader import DorabotLogReader
from piplan.legacy_expert.expert_dataset_converter import ExpertDatasetConverter


class LegacyExpertPolicyAdapter:
    """Offline adapter for loading legacy expert demonstrations as pi0.5 examples."""

    def __init__(self, include_images: bool = True, action_source: str = "expert"):
        self.reader = DorabotLogReader()
        self.converter = ExpertDatasetConverter(include_images=include_images, action_source=action_source)

    def load_examples(self, patterns: list[str]) -> list[dict]:
        return self.converter.convert(self.reader.read(patterns))
