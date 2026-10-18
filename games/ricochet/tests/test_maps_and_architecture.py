import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ricochet.game.config import RulesConfig
from ricochet.game.maps import available_maps, load_map, validate_map
from ricochet.game.types import MapDefinition, Vec2


def test_three_maps_load_and_are_symmetric() -> None:
    assert available_maps() == ("channels", "classic", "crossfire")
    for map_id in available_maps():
        definition = load_map(map_id)
        assert definition.width == 960
        assert definition.height == 640
        first, second = sorted(definition.spawns, key=lambda spawn: spawn.player_id)
        assert first.position.x == definition.width - second.position.x
        assert first.position.y == second.position.y


@pytest.mark.parametrize(
    ("problem", "message"), [("not finite", "finite"), ("spawn in a wall", "overlaps a wall")]
)
def test_invalid_maps_are_refused(problem: str, message: str) -> None:
    definition = load_map("classic")
    first, second = definition.spawns
    broken = {
        "not finite": replace(definition, width=math.nan),
        # The centre of Classic's centre block.
        "spawn in a wall": replace(
            definition, spawns=(replace(first, position=Vec2(480, 320)), second)
        ),
    }[problem]
    with pytest.raises(ValueError, match=message):
        validate_map(broken, RulesConfig())


def test_map_serialization_round_trips_and_normalizes_numbers() -> None:
    definition = load_map("crossfire")
    assert MapDefinition.from_dict(definition.to_dict()) == definition
    rebuilt = MapDefinition.from_dict(
        definition.to_dict() | {"width": 960, "height": 640}  # Integers, as in map files.
    )
    assert rebuilt == definition and isinstance(rebuilt.width, float)


def test_what_the_arena_runs_needs_no_pygame_numpy_or_gymnasium() -> None:
    # What arena_game.py imports, and every bot it loads, in tournaments and in the browser.
    code = (
        "import sys\n"
        "import ricochet.game.engine, ricochet.game.mapgen, ricochet.themes\n"
        "from ricochet.runtime.loading import available_ais, load_controller\n"
        "for name in available_ais():\n"
        "    load_controller(name)\n"
        "print(sorted({'pygame', 'numpy', 'gymnasium'} & set(sys.modules)))\n"
    )
    game = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=game, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
