import math

import pytest

from ricochet.bot import find_path
from ricochet.game.engine import GameEngine
from ricochet.game.mapgen import GeneratorConfig, generate_map
from ricochet.game.maps import load_map
from ricochet.runtime.cli import build_parser

# A few seeds, including every one the Arena plays (GENERATED in arena_game.py).
SEEDS = (0, 1, 2, 3, 8, 12, 18, 21)


def _rects(definition):
    return [(w.left, w.top, w.right, w.bottom) for w in definition.walls]


def _images(definition):
    """The map's walls and spawns under mirror and half-turn symmetry."""
    width, height = definition.width, definition.height
    spawns = [spawn.position for spawn in definition.spawns]
    mirror = (
        [(width - r, t, width - left, b) for left, t, r, b in _rects(definition)],
        [(width - p.x, p.y) for p in spawns],
    )
    turn = (
        [(width - r, height - b, width - left, height - t) for left, t, r, b in _rects(definition)],
        [(width - p.x, height - p.y) for p in spawns],
    )
    return mirror, turn


def _near_symmetric(definition, tolerance):
    rects = _rects(definition)
    spawns = [(spawn.position.x, spawn.position.y) for spawn in definition.spawns]
    for walls, mirrored_spawns in _images(definition):
        walls_match = all(
            any(
                max(abs(a - b) for a, b in zip(rect, image, strict=True)) <= tolerance
                for image in walls
            )
            for rect in rects
        )
        # Each spawn's image lies near the other spawn.
        spawns_match = all(
            math.dist(spawns[1 - index], image) <= tolerance
            for index, image in enumerate(mirrored_spawns)
        )
        if walls_match and spawns_match:
            return True
    return False


def test_generation_is_deterministic_and_varied() -> None:
    assert generate_map(7) == generate_map(7)
    assert len({generate_map(seed).walls for seed in SEEDS}) == len(SEEDS)
    assert load_map("generated:7") == generate_map(7)
    assert generate_map(7).id == "generated:7"


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_maps_are_playable(seed: int) -> None:
    engine = GameEngine(generate_map(seed))  # Runs the ordinary map validation.
    observation = engine.observation_for(0)
    me, opponent = observation.me.position, observation.opponent.position
    radius = engine.rules.tank_radius
    route = find_path(observation, me, opponent, radius, spacing=10, max_nodes=10_000)
    assert route and math.dist((route[-1].x, route[-1].y), (opponent.x, opponent.y)) < 10


@pytest.mark.parametrize("seed", SEEDS)
def test_maps_are_slightly_disturbed_symmetric_layouts(seed: int) -> None:
    symmetric = generate_map(seed, GeneratorConfig(asymmetry=0))
    assert _near_symmetric(symmetric, tolerance=0)
    disturbed = generate_map(seed)
    # Every free edge and spawn coordinate moves at most 16 px, so images differ by <= 32.
    assert _near_symmetric(disturbed, tolerance=32)
    assert not _near_symmetric(disturbed, tolerance=0)


def test_the_command_line_takes_generated_maps() -> None:
    parser = build_parser()
    assert parser.parse_args(["match", "--map", "generated:3"]).map == "generated:3"
    for bad in ("generated:x", "nowhere"):
        with pytest.raises(SystemExit):
            parser.parse_args(["match", "--map", bad])
