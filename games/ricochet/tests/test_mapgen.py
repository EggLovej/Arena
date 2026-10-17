import math

import pytest

from ricochet.bot import find_path
from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.mapgen import GeneratorConfig, generate_map
from ricochet.game.maps import load_map
from ricochet.runtime.cli import build_parser
from ricochet.runtime.match import run_headless

SEEDS = range(40)


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
    rules, config = RulesConfig(), GeneratorConfig()
    definition = generate_map(seed)
    engine = GameEngine(definition)  # Runs the ordinary map validation.
    gap = 2 * rules.tank_radius + config.passage_clearance
    assert 1 <= len(definition.walls) <= rules.max_walls
    for wall in definition.walls:
        assert min(wall.width, wall.height) >= config.min_thickness
    rects = _rects(definition)
    for index, first in enumerate(rects):
        for second in rects[index + 1 :]:
            dx = max(0, second[0] - first[2], first[0] - second[2])
            dy = max(0, second[1] - first[3], first[1] - second[3])
            assert not 0 < math.hypot(dx, dy) < gap, (first, second)
    observation = engine.observation_for(0)
    me, opponent = observation.me.position, observation.opponent.position
    route = find_path(observation, me, opponent, rules.tank_radius, spacing=10, max_nodes=10_000)
    assert route and math.dist((route[-1].x, route[-1].y), (opponent.x, opponent.y)) < 10


@pytest.mark.parametrize("seed", SEEDS)
def test_maps_are_slightly_disturbed_symmetric_layouts(seed: int) -> None:
    symmetric = generate_map(seed, GeneratorConfig(asymmetry=0))
    assert _near_symmetric(symmetric, tolerance=0)
    disturbed = generate_map(seed)
    # Every free edge and spawn coordinate moves at most 16 px, so images differ by <= 32.
    assert _near_symmetric(disturbed, tolerance=32)


def test_disturbance_breaks_exact_symmetry() -> None:
    assert not any(_near_symmetric(generate_map(seed), tolerance=0) for seed in SEEDS)


def test_generated_maps_work_in_matches_and_cli() -> None:
    summary = run_headless(
        "direct_ai", "bank_shot_ai", "generated:3", rules=RulesConfig(max_seconds=1)
    )
    assert summary.map_id == "generated:3" and summary.faults == (0, 0)
    parser = build_parser()
    assert parser.parse_args(["match", "--map", "generated:3"]).map == "generated:3"
    for bad in ("generated:x", "nowhere"):
        with pytest.raises(SystemExit):
            parser.parse_args(["match", "--map", bad])
