"""Small CPU PPO baseline with frozen opponents and explicit run lineage."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ricochet.game.config import RULES_VERSION, RulesConfig
from ricochet.runtime.loading import load_controller
from ricochet.training.checkpoints import CheckpointController, metadata_path
from ricochet.training.encoding import ACTION_VERSION, ENCODER_VERSION
from ricochet.training.evaluation import SCRIPTED, TEST_MAPS, TRAINING_MAPS, evaluate
from ricochet.training.gym_env import RicochetEnv


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    total_timesteps: int = 1024000
    n_envs: int = 4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 5
    learning_rate: float = 0.0003
    gamma: float = 0.999
    gae_lambda: float = 0.995
    clip_range: float = 0.2
    target_kl: float = 0.02
    ent_coef: float = 0.005
    snapshot_interval: int = 102400
    smoke: bool = False


class OpponentPool:
    def __init__(self):
        self.paths = []
        self.cache = {}
        self.current = None
        self.last_name = None

    def reset(self, context):
        rng = np.random.default_rng(context.seed)
        if self.paths and rng.random() < 0.5:
            name = str(self.paths[int(rng.integers(len(self.paths)))])
            if name not in self.cache:
                self.cache[name] = CheckpointController(name)
        else:
            name = SCRIPTED[int(rng.integers(len(SCRIPTED)))]
            if name not in self.cache:
                self.cache[name] = load_controller(name)
        self.last_name = name
        self.current = self.cache[name]
        self.current.reset(context)

    def act(self, observation):
        return self.current.act(observation)

    def update(self, paths):
        self.paths = list(paths[-5:])
        self.cache = {
            k: v
            for k, v in self.cache.items()
            if k in SCRIPTED or k in {str(p) for p in self.paths}
        }


def code_identity():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {"git_revision": revision, "source_sha256": digest.hexdigest()}


def train(output: Path, config: TrainConfig, resume: Path | None = None):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)
    if config.total_timesteps <= 0 or config.n_envs <= 0 or config.n_steps <= 0:
        raise ValueError("training lengths must be positive")
    if config.total_timesteps % (config.n_envs * config.n_steps):
        raise ValueError("steps must be a multiple of n_envs * n_steps")
    if config.snapshot_interval % (config.n_envs * config.n_steps):
        raise ValueError("checkpoint interval must be a multiple of rollout size")
    if resume:
        prior = json.loads(metadata_path(resume).read_text())
        for key, value in asdict(config).items():
            if key not in {"seed", "total_timesteps", "smoke", "snapshot_interval"}:
                if prior["config"][key] != value:
                    raise ValueError(
                        f"resume configuration changes {key}; start a new policy instead"
                    )
    if output.exists():
        raise ValueError("Choose a new run directory; existing runs are immutable")
    output.mkdir(parents=True)
    rules = RulesConfig()
    manifest = {
        "config": asdict(config),
        "rules": rules.to_dict(),
        "rules_version": RULES_VERSION,
        "encoder_version": ENCODER_VERSION,
        "action_version": ACTION_VERSION,
        "code": code_identity(),
        "parent_checkpoint": str(resume.resolve()) if resume else None,
        "parent_checkpoint_sha256": hashlib.sha256(resume.read_bytes()).hexdigest()
        if resume
        else None,
        "resume_boundary": "fresh episodes; optimizer/model/counters retained",
        "maps": list(TRAINING_MAPS),
        "test_maps": list(TEST_MAPS),
        "scripted_opponents": list(SCRIPTED),
        "dependencies": {
            name: version(name) for name in ("torch", "stable-baselines3", "numpy", "gymnasium")
        },
        "map_definitions": {},
        "opponent_identities": {},
    }
    from ricochet.game.maps import load_map
    from ricochet.runtime.loading import default_ais_directory

    for name in TRAINING_MAPS + TEST_MAPS:
        manifest["map_definitions"][name] = load_map(name).to_dict()
    for name in SCRIPTED:
        bot_path = default_ais_directory() / f"{name}.py"
        manifest["opponent_identities"][name] = hashlib.sha256(bot_path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pools = [OpponentPool() for _ in range(config.n_envs)]
    envs = DummyVecEnv(
        [
            lambda pool=pool: Monitor(
                RicochetEnv(maps=TRAINING_MAPS, opponent=lambda: pool, jitter_probability=0.5)
            )
            for pool in pools
        ]
    )
    envs.seed(config.seed)
    if resume:
        CheckpointController(resume)  # Validate versions before accepting optimizer/model state.
        model = PPO.load(resume, env=envs, device="cpu")
        model.set_random_seed(config.seed)
    else:
        model = PPO(
            "MlpPolicy",
            envs,
            device="cpu",
            seed=config.seed,
            verbose=0,
            policy_kwargs={
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                "activation_fn": torch.nn.Tanh,
            },
            **{
                k: getattr(config, k)
                for k in (
                    "n_steps",
                    "batch_size",
                    "n_epochs",
                    "learning_rate",
                    "gamma",
                    "gae_lambda",
                    "clip_range",
                    "target_kl",
                    "ent_coef",
                )
            },
        )
    snapshots = []
    if resume:
        snapshots = [resume.resolve()]
        for pool in pools:
            pool.update(snapshots)
    parent_steps = model.num_timesteps
    interval = config.n_envs * config.n_steps if config.smoke else config.snapshot_interval
    started = time.perf_counter()

    class Recorder(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last_checkpoint = parent_steps
            self.best = -float("inf")
            self.components = {key: 0.0 for key in ("damage_dealt", "damage_taken", "outcome")}
            self.episodes = 0
            self.outcomes = {"wins": 0, "draws": 0, "losses": 0}
            self.episode_records = []

        def _on_step(self):
            for info in self.locals["infos"]:
                if info.get("opponent_faults", 0):
                    raise RuntimeError("training opponent faulted; refusing to learn from forfeits")
                for key, value in info.get("reward_components", {}).items():
                    self.components[key] += float(value)
                if "episode" in info:
                    self.episodes += 1
                    winner = info["winner"]
                    outcome = (
                        "draws"
                        if winner is None
                        else "wins"
                        if winner == info["player_id"]
                        else "losses"
                    )
                    self.outcomes[outcome] += 1
                    self.episode_records.append(
                        {
                            "episode_seed": info["episode_seed"],
                            "map_id": info["map_id"],
                            "player_id": info["player_id"],
                            "opponent": info.get("opponent_name"),
                            "outcome": outcome,
                        }
                    )
            return True

        def save(self):
            path = output / f"checkpoint_{self.num_timesteps:09d}.zip"
            if path.exists():
                return
            self.model.save(path)
            meta = {
                **manifest,
                "steps": self.num_timesteps,
                "opponent_snapshots": [
                    {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in snapshots
                ],
            }
            metadata_path(path).write_text(json.dumps(meta, indent=2) + "\n")
            snapshots.append(path.resolve())
            for pool in pools:
                pool.update(snapshots)
            result = evaluate(CheckpointController(path), quick=config.smoke)
            (output / f"eval_{self.num_timesteps:09d}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            for alias in ("latest", "best"):
                if alias == "best" and (result["score"] is None or result["score"] <= self.best):
                    continue
                shutil.copyfile(path, output / f"{alias}.zip")
                shutil.copyfile(metadata_path(path), output / f"{alias}.json")
            if result["score"] is not None:
                self.best = max(self.best, result["score"])
            row = {
                "steps": self.num_timesteps,
                "episodes": self.episodes,
                "outcomes": dict(self.outcomes),
                "wall_seconds": time.perf_counter() - started,
                "transitions_per_second": (self.num_timesteps - parent_steps)
                / (time.perf_counter() - started),
                "reward_components": self.components,
                "validation_score": result["score"],
                "opponents": [pool.last_name for pool in pools],
                "metrics": {
                    k: float(v)
                    for k, v in self.model.logger.name_to_value.items()
                    if isinstance(v, (int, float, np.number))
                },
            }
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            with (output / "episodes.jsonl").open("a") as handle:
                for episode in self.episode_records:
                    handle.write(json.dumps(episode) + "\n")
            self.episode_records.clear()
            self.last_checkpoint = self.num_timesteps
            print(
                f"checkpoint {self.num_timesteps}: validation score {result['score']}", flush=True
            )

        def _on_rollout_start(self):
            if self.num_timesteps - self.last_checkpoint >= interval:
                self.save()

        def _on_training_end(self):
            self.save()

    callback = Recorder()
    try:
        model.learn(config.total_timesteps, callback=callback, reset_num_timesteps=resume is None)
    finally:
        envs.close()
    return output / "latest.zip"
