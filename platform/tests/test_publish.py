import json

from arena import publish
from arena.tournament import run, schedule


def test_published_files_and_rederived_tables(race, tmp_path):
    plans = schedule(race)
    finished = list(run("race", plans, 1))
    rows = [row for _, row, _ in finished]
    replays = {plan.match_id: replay for plan, _, replay in finished if replay}
    publish.publish_game(race, tmp_path)
    publish.publish_results(race, tmp_path, rows, replays, "abc123", {"bots": None}, 1.5)

    games = json.loads((tmp_path / "games.json").read_text())
    assert games == [{"id": "race", "name": "Race", "tagline": "Run to the goal.", "seats": ["red", "blue", "green"], "maps": 2, "bots": 6}]
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
