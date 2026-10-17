import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.training.encoding import OBSERVATION_SIZE, decode_action, encode_observation
from ricochet.training.gym_env import RicochetEnv


def test_encoder_has_fixed_normalized_shape() -> None:
    engine = GameEngine(load_map("channels"))
    vector = encode_observation(engine.observation_for(0))
    assert vector.shape == (OBSERVATION_SIZE,)
    assert vector.dtype == np.float32
    assert np.all(vector >= -1) and np.all(vector <= 1)


def test_action_decoder_thresholds_triggers() -> None:
    action = decode_action(np.array([1, 1, -1, 0, 0.1, -0.1], dtype=np.float32))
    assert action.move.length() == pytest.approx(1)
    assert action.fire
    assert not action.shield


def test_seeded_resets_are_identical() -> None:
    env = RicochetEnv(seat_mode="random")
    first, first_info = env.reset(seed=42)
    second, second_info = env.reset(seed=42)
    np.testing.assert_array_equal(first, second)
    assert first_info["player_id"] == second_info["player_id"]
    assert first_info["state_hash"] == second_info["state_hash"]


def test_time_limit_is_terminal() -> None:
    env = RicochetEnv(rules=RulesConfig(max_seconds=1 / 30), seat_mode=0)
    env.reset(seed=0)
    _observation, _reward, terminated, truncated, _info = env.step(np.zeros(6, dtype=np.float32))
    assert terminated
    assert not truncated


def test_gymnasium_contract() -> None:
    check_env(RicochetEnv(rules=RulesConfig(max_seconds=0.1)), skip_render_check=True)
