"""Helpers that only the tests need."""

import copy
import functools
from dataclasses import fields

import numpy as np
import torch

from game import Game
from game_types import BLUE, RED, MapDefinition, SendAction
from loading import available_maps, load_ai, load_map
from ml.encoding.action_encoder import LegalActions, legal_actions
from ml.encoding.state_encoder import EncodedState, encode_state
from ml.map_generator import generate_map
from ml.models import ModelInput
from ml.models.interface import ModelOutput
from ml.play.rollout import WAIT, GameRecord, Trajectory


def trajectory_input(
    trajectory: Trajectory,
    decisions: slice | np.ndarray = slice(None),
    device: torch.device | str = "cpu",
) -> ModelInput:
    """Recorded decisions as a batch the model can be run on again."""

    base_features = torch.from_numpy(trajectory.base_features[decisions])
    count, bases = base_features.shape[:2]
    limits = None if trajectory.route_limits is None else trajectory.limit_table(decisions).astype(np.float32)
    return ModelInput(
        base_features=base_features,
        pair_features=torch.tensor(trajectory.pair_features).expand(count, -1, -1, -1),
        route_features=torch.from_numpy(trajectory.route_table(decisions)),
        global_features=torch.from_numpy(trajectory.global_features[decisions]),
        base_mask=torch.ones(count, bases, dtype=torch.bool),
        allowed=torch.from_numpy(trajectory.allowed(decisions)),
        most=None if limits is None else torch.from_numpy(limits[..., 0].copy()),
        needed=None if limits is None else torch.from_numpy(limits[..., 1].copy()),
    ).to(device)


def trajectory_choices(
    trajectory: Trajectory,
    decisions: slice | np.ndarray = slice(None),
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """The recorded choices in the form ``ml.models.policy`` works with (units, for free amounts)."""

    source = trajectory.source[decisions].astype(np.int64)
    target = trajectory.target[decisions].astype(np.int64)
    amount = trajectory.amount[decisions].astype(np.int64)
    waited = source == WAIT
    route = np.where(waited, 0, 1 + source * len(trajectory.base_ids) + target)
    return (
        torch.from_numpy(route).to(device),
        torch.from_numpy(np.where(waited, 0, amount)).to(device),
    )


def repeated(output: ModelOutput, row: int, count: int) -> ModelOutput:
    """One position's output ``count`` times over, to draw from it many times at once."""

    parts = {field.name: getattr(output, field.name) for field in fields(output)}
    return ModelOutput(**{name: None if part is None else part[row : row + 1].expand(count, *part.shape[1:]) for name, part in parts.items()})


def replay(record: GameRecord) -> Game:
    """Play a recorded game again from its list of moves, and return the finished game.

    Fails if the game refuses any of the moves.
    """

    game = Game(record.matchup.map)
    moves: dict[int, dict[int, SendAction]] = {}
    for tick, seat, source_id, target_id, units in record.actions:
        moves.setdefault(tick, {})[seat] = SendAction(source_id, target_id, units)
    while game.tick < record.ticks:
        refused = game.step(moves.get(game.tick, {}))
        assert not refused, f"tick {game.tick}: {refused}"
    return game


@functools.cache
def two_player_maps() -> tuple[MapDefinition, ...]:
    """Generated maps 0 to 7 and every hand-made map for two players."""

    made = (load_map(name) for name in available_maps())
    return tuple(generate_map(seed).definition for seed in range(8)) + tuple(
        definition for definition in made if Game(definition).players == (RED, BLUE)
    )


@functools.cache
def _scripted_match(definition: MapDefinition) -> tuple[Game, ...]:
    """Strategic AI (Red) against Aggressive AI (Blue): the game at every 5th tick and at its end, up to tick 600.

    Played once per map and kept, as many tests look at the same positions.
    """

    strategic, aggressive = load_ai("strategic_ai"), load_ai("aggressive_ai")
    game, kept = Game(definition), []
    while True:
        if game.tick % 5 == 0 or game.is_finished():
            kept.append(copy.deepcopy(game))
        if game.is_finished() or game.tick == 600:
            return tuple(kept)
        game.step({RED: strategic(game.view_for(RED)), BLUE: aggressive(game.view_for(BLUE))})


def scripted_games(definition: MapDefinition, every: int, until: int = 500) -> list[Game]:
    """Strategic AI (Red) against Aggressive AI (Blue) on a map: copies of the game, free to play on,
    every ``every`` ticks (a multiple of 5) from the start, before tick ``until`` and while it lasts."""

    match = _scripted_match(definition)
    return [copy.deepcopy(game) for game in match if game.tick % every == 0 and game.tick < until and not game.is_finished()]


def scripted_game(definition: MapDefinition, tick: int) -> Game:
    """A copy of that game at ``tick`` (a multiple of 5, up to 600), or at its end if it ends before."""

    match = _scripted_match(definition)
    return copy.deepcopy(next((game for game in match if game.tick >= tick), match[-1]))


def positions(seeds=range(6), every: int = 40, ticks: int = 200) -> tuple[list[EncodedState], list[LegalActions]]:
    """Encoded positions from generated maps of different sizes: that game as Red sees it every
    ``every`` ticks (a multiple of 5), up to tick ``ticks``."""

    views = [
        game.view_for(RED)
        for seed in seeds
        for game in _scripted_match(generate_map(seed).definition)
        if 0 < game.tick <= ticks and game.tick % every == 0
    ]
    return [encode_state(view) for view in views], [legal_actions(view) for view in views]
