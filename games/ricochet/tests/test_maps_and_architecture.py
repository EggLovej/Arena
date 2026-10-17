from pathlib import Path

import pytest

from ricochet.game.config import RulesConfig
from ricochet.game.maps import available_maps, load_map, validate_map
from ricochet.game.types import MapDefinition, SpawnDefinition, Vec2, WallDefinition


def test_three_maps_load_and_are_symmetric() -> None:
    assert available_maps() == ("channels", "classic", "crossfire")
    for map_id in available_maps():
        definition = load_map(map_id)
        assert definition.width == 960
        assert definition.height == 640
        first, second = sorted(definition.spawns, key=lambda spawn: spawn.player_id)
        assert first.position.x == definition.width - second.position.x
        assert first.position.y == second.position.y


def test_map_rejects_spawn_overlapping_wall() -> None:
    definition = MapDefinition(
        id="bad",
        name="Bad",
        width=100,
        height=100,
        spawns=(
            SpawnDefinition(0, Vec2(30, 50), Vec2(1, 0)),
            SpawnDefinition(1, Vec2(80, 50), Vec2(-1, 0)),
        ),
        walls=(WallDefinition("wall", 20, 40, 20, 20),),
    )
    with pytest.raises(ValueError, match="overlaps"):
        validate_map(definition, RulesConfig(tank_radius=10))


def test_game_package_does_not_import_pygame_or_gymnasium() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "ricochet" / "game"
    text = "\n".join(path.read_text() for path in root.glob("*.py"))
    assert "import pygame" not in text
    assert "import gymnasium" not in text
