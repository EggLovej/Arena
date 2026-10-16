import json

import pytest

from arena import publish
from arena.registry import repo_root
from arena.tournament import run, schedule
from conftest import FIXTURES


def test_published_files_and_rederived_tables(race, tmp_path):
    plans = schedule(race)
    finished = list(run("race", plans, 1))
    rows = [row for _, row, _ in finished]
    replays = {plan.match_id: replay for plan, _, replay in finished if replay}
    publish.publish_game(race, tmp_path)
    publish.publish_results(race, tmp_path, rows, replays, "abc123", {"bots": None}, 1.5)

    games = json.loads((tmp_path / "games.json").read_text())
    assert games == [{"id": "race", "name": "Race", "tagline": "Run to the goal.", "seats": ["red", "blue", "green"], "maps": 2, "bots": 8}]
    game = json.loads((tmp_path / "race" / "game.json").read_text())
    assert [entry["id"] for entry in game["maps"]] == ["short", "long"]
    assert game["maps"][0]["start"] == {"runners": {"red": [0], "blue": [0]}}
    assert game["rating_anchor"] == ["steady", 1000.0]

    folder = tmp_path / "race"
    assert sorted(path.stem for path in (folder / "replays").glob("*.json")) == sorted(replays)
    matches = json.loads((folder / "matches.json").read_text())
    assert len(matches) == len(rows) and sum(row["replay"] for row in matches) == len(replays)
    leaderboard = json.loads((folder / "leaderboard.json").read_text())
    assert leaderboard["inputs"] == "abc123" and leaderboard["matches"] == len(rows)
    assert publish.published_inputs(race, tmp_path) == "abc123"

    # Tables come from results.json alone, so they can be redone without playing again.
    before = (folder / "leaderboard.json").read_text()
    (folder / "leaderboard.json").unlink()
    (folder / "matches.json").unlink()
    assert publish.publish_tables(race, tmp_path)
    assert (folder / "leaderboard.json").read_text() == before


def test_no_tables_without_a_tournament(race, tmp_path):
    assert not publish.publish_tables(race, tmp_path)
    assert publish.published_inputs(race, tmp_path) is None


def test_inputs_change_with_code_but_not_with_notes(race, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "games" / "race" / "tests").mkdir(parents=True)
    (root / "platform" / "arena").mkdir(parents=True)
    (root / "games" / "race" / "arena_game.py").write_text("x = 1\n")
    (root / "platform" / "arena" / "runner.py").write_text("y = 1\n")
    monkeypatch.setenv("ARENA_ROOT", str(root))
    settings = {"bots": None}
    original = publish.inputs_fingerprint(race, settings)

    (root / "games" / "race" / "NOTES.md").write_text("notes")
    (root / "games" / "race" / "tests" / "test_x.py").write_text("assert True")
    (root / "platform" / "arena" / "ratings.py").write_text("z = 2\n")
    assert publish.inputs_fingerprint(race, settings) == original

    assert publish.inputs_fingerprint(race, {"bots": ["steady"]}) != original
    (root / "games" / "race" / "arena_game.py").write_text("x = 2\n")
    assert publish.inputs_fingerprint(race, settings) != original

    before = publish.inputs_fingerprint(race, settings)
    (root / "pyproject.toml").write_text('[project]\nname = "arena"\n[project.optional-dependencies]\nneural = ["torch==1"]\n')
    pinned = publish.inputs_fingerprint(race, settings)
    assert pinned != before
    (root / "pyproject.toml").write_text('[project]\nname = "arena"\n[project.optional-dependencies]\nneural = ["torch==2"]\n')
    assert publish.inputs_fingerprint(race, settings) != pinned
    (root / "pyproject.toml").write_text('[project]\nname = "arena"\nversion = "9"\n[project.optional-dependencies]\nneural = ["torch==1"]\n')
    assert publish.inputs_fingerprint(race, settings) == pinned


def test_inputs_need_the_game_files(race, tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="is ARENA_ROOT right"):
        publish.inputs_fingerprint(race, {})


def test_a_relative_arena_root_stays_where_it_started(race, monkeypatch):
    monkeypatch.chdir(FIXTURES.parent)
    monkeypatch.setenv("ARENA_ROOT", FIXTURES.name)
    assert repo_root() == FIXTURES
    monkeypatch.chdir(FIXTURES / "games" / "race")  # as loading a game can
    assert repo_root() == FIXTURES
