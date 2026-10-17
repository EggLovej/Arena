"""Local, trusted PPO checkpoints exposed through the ordinary controller API."""

import json
from pathlib import Path

from ricochet.bot import BaseController
from ricochet.game.config import RULES_VERSION
from ricochet.training.encoding import (
    ACTION_VERSION,
    ENCODER_VERSION,
    decode_action,
    encode_observation,
)


def metadata_path(path):
    return Path(path).with_suffix(".json")


class CheckpointController(BaseController):
    def __init__(self, path):
        from stable_baselines3 import PPO

        self.path = Path(path).resolve()
        self.metadata = json.loads(metadata_path(self.path).read_text())
        expected = {
            "rules_version": RULES_VERSION,
            "encoder_version": ENCODER_VERSION,
            "action_version": ACTION_VERSION,
        }
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError(f"incompatible checkpoint {key}")
        self.model = PPO.load(self.path, device="cpu")

    def reset(self, context):
        if context.rules.to_dict() != self.metadata["rules"]:
            raise ValueError("checkpoint rules differ from match rules")
        super().reset(context)

    def act(self, observation):
        action, _ = self.model.predict(
            encode_observation(observation, self.rules), deterministic=True
        )
        return decode_action(action)
