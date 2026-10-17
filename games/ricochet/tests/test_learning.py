"""Cheap model serialization/pool integration checks; no long training runs."""

import json

import numpy as np
import pytest

from ricochet.game.config import RULES_VERSION, RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.game.types import MatchContext
from ricochet.training.encoding import ACTION_VERSION, ENCODER_VERSION


def test_checkpoint_pool_and_version_contract(tmp_path):
    import torch
    from stable_baselines3 import PPO

    from ricochet.training.checkpoints import CheckpointController, metadata_path
    from ricochet.training.gym_env import RicochetEnv
    from ricochet.training.train import OpponentPool

    torch.set_num_threads(1)
    model = PPO("MlpPolicy", RicochetEnv(), n_steps=8, batch_size=8, device="cpu")
    path = tmp_path / "policy.zip"
    model.save(path)
    metadata = {
        "rules_version": RULES_VERSION,
        "encoder_version": ENCODER_VERSION,
        "action_version": ACTION_VERSION,
        "rules": RulesConfig().to_dict(),
    }
    metadata_path(path).write_text(json.dumps(metadata))
    controller = CheckpointController(path)
    context = MatchContext("classic", 0, 1, 0, 30, 2700, RulesConfig())
    controller.reset(context)
    observation = GameEngine(load_map("classic")).observation_for(0)
    first = controller.act(observation)
    assert first == controller.act(observation)
    pool = OpponentPool()
    pool.update([path])
    names = set()
    for seed in range(20):
        pool.reset(MatchContext("classic", 0, 1, seed, 30, 2700, RulesConfig()))
        pool.act(observation)
        names.add(pool.last_name)
    assert str(path) in names and len(names) > 1
    with pytest.raises(ValueError, match="rules"):
        controller.reset(
            MatchContext("classic", 0, 1, 0, 30, 2700, RulesConfig.preset("legacy_shield"))
        )
    metadata["encoder_version"] = 999
    metadata_path(path).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="encoder"):
        CheckpointController(path)


def test_tiny_training_resume_and_finite_metrics(tmp_path, monkeypatch):
    from ricochet.training import train as training
    from ricochet.training.checkpoints import metadata_path

    # Evaluate policies separately above; here isolate optimizer persistence and callbacks.
    monkeypatch.setattr(training, "evaluate", lambda *a, **k: {"score": 0.5, "games": 0})
    config = training.TrainConfig(
        n_envs=1,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        total_timesteps=32,
        snapshot_interval=16,
        smoke=True,
    )
    first = training.train(tmp_path / "first", config)
    second = training.train(tmp_path / "resumed", config, first)
    assert json.loads(metadata_path(second).read_text())["steps"] == 64
    from stable_baselines3 import PPO

    model = PPO.load(second)
    assert model._n_updates == 4
    for row in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines():
        data = json.loads(row)
        assert all(np.isfinite(v) for v in data["metrics"].values())
    with pytest.raises(ValueError, match="immutable"):
        training.train(tmp_path / "first", config)
    manifest = json.loads((tmp_path / "resumed" / "manifest.json").read_text())
    assert manifest["parent_checkpoint_sha256"]
    assert manifest["map_definitions"]["classic"]["walls"]
    assert manifest["code"]["source_sha256"]
