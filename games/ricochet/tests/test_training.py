import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine, _ProjectileState
from ricochet.game.maps import load_map
from ricochet.game.types import Vec2
from ricochet.training.encoding import OBSERVATION_SIZE, decode_action, encode_observation
from ricochet.training.gym_env import RicochetEnv


def test_encoder_has_fixed_normalized_shape_and_capacity() -> None:
    engine = GameEngine(load_map("channels"))
    vector = encode_observation(engine.observation_for(0))
    assert vector.shape == (281,) == (OBSERVATION_SIZE,)
    assert vector.dtype == np.float32
    assert np.all(vector >= -1) and np.all(vector <= 1)
    with pytest.raises(ValueError, match="capacity"):  # More shots than the encoder has slots.
        RicochetEnv(rules=RulesConfig(max_projectiles=25))


def test_action_decoder_thresholds_triggers() -> None:
    action = decode_action(np.array([1, 1, -1, 0, 0.1, -0.1], dtype=np.float32))
    assert action.move.length() == pytest.approx(1)
    assert action.fire
    assert not action.shield


def test_seeded_resets_repeat_and_later_resets_advance() -> None:
    env = RicochetEnv(maps=("classic", "crossfire"), jitter_probability=0.5)

    def episodes():
        resets = [env.reset(seed=42), *(env.reset() for _ in range(4))]
        return [(obs.tolist(), info["episode_seed"], info["player_id"]) for obs, info in resets]

    first = episodes()
    assert len({seed for _, seed, _ in first}) == 5
    assert episodes() == first


def test_the_clock_terminates_and_an_external_cutoff_truncates() -> None:
    env = RicochetEnv(rules=RulesConfig(max_seconds=1 / 30), seat_mode=0)
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
    assert terminated and not truncated
    env = RicochetEnv(max_episode_steps=1)
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
    assert truncated and not terminated


def test_hitting_yourself_costs_reward() -> None:
    env = RicochetEnv(opponent="template_ai", seat_mode=0)
    env.reset(seed=0)
    p = env.engine.tanks[0].position
    env.engine.projectiles = [_ProjectileState(99, 0, Vec2(p.x - 30, p.y), Vec2(420, 0), True, 5)]
    _, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
    assert reward == pytest.approx(-0.1)


def test_gymnasium_contract() -> None:
    check_env(RicochetEnv(rules=RulesConfig(max_seconds=0.1)), skip_render_check=True)


def test_parallel_contract() -> None:
    from pettingzoo.test import parallel_api_test

    from ricochet.training.parallel_env import RicochetParallelEnv

    parallel_api_test(RicochetParallelEnv(rules=RulesConfig(max_seconds=0.1)), num_cycles=20)
