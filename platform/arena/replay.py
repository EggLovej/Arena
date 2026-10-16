"""Read replays and check that they play back exactly as recorded.

A replay (written by ``arena.runner.play_match``) holds:

- a header: game, map, seed, tick length, the seats and their bots;
- ``board``: what never changes during the match, for drawing;
- ``frames``: a snapshot of every tick as keyframes and deltas (``arena.frames``);
- ``actions``: every action a bot chose, as ``[tick, seat, action]``;
- ``hashes``: a hash of the exact state at every keyframe and at the end;
- ``outcome`` and ``reports``: the result and how each bot behaved.

The rules are deterministic, so replaying the actions on a new match must
reproduce every hash, keyframe and the result. ``verify`` checks exactly that.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

from arena.contract import Game, Outcome, state_hash
from arena.runner import REPLAY_FORMAT, REPLAY_VERSION


def write_replay(path: Path, replay: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(replay, separators=(",", ":"))
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode(), mtime=0))
    else:
        path.write_text(text)
    return path


def read_replay(path: Path) -> dict:
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    replay = json.loads(data)
    if replay.get("format") != REPLAY_FORMAT:
        raise ValueError(f"{path} is not an arena replay")
    if replay.get("version") != REPLAY_VERSION:
        raise ValueError(f"{path} is replay version {replay.get('version')}, expected {REPLAY_VERSION}")
    return replay


def verify(game: Game, replay: dict) -> list[str]:
    """Play the recorded actions again; returns what differs (an empty list if nothing)."""

    problems = []
    if replay["game"] != game.info.id:
        return [f"replay is for {replay['game']}, not {game.info.id}"]
    if replay.get("fingerprint") not in (None, game.fingerprint()):
        problems.append("the game's rules, maps or bots changed since this replay was recorded")

    actions_by_tick: dict[int, dict[str, object]] = defaultdict(dict)
    for tick, seat, action in replay["actions"]:
        actions_by_tick[tick][seat] = game.action_from_json(action)

    match = game.new_match(replay["map"], replay["seed"])
    frames = replay["frames"]
    keyframe_every = frames["keyframe_every"]
    final_tick = frames["count"] - 1

    def check(tick: int) -> None:
        if tick % keyframe_every:
            return
        expected_hash = replay["hashes"].get(str(tick))
        if expected_hash is not None and expected_hash != state_hash(match.save_state()):
            problems.append(f"state differs at tick {tick}")
        expected_frame = frames["keyframes"].get(str(tick))
        if expected_frame is not None and _plain(expected_frame) != _plain(match.snapshot()):
            problems.append(f"snapshot differs at tick {tick}")

    check(0)
    while match.tick < final_tick:
        match.step(actions_by_tick.get(match.tick, {}))
        check(match.tick)

    if replay["hashes"].get("final") != state_hash(match.save_state()):
        problems.append("final state differs")
    if Outcome.from_json(replay["outcome"]) != match.outcome():
        problems.append(f"outcome differs: recorded {replay['outcome']}, replayed {match.outcome().to_json()}")
    return problems


def _plain(value):
    """The value as it reads back from JSON (tuples become lists)."""

    return json.loads(json.dumps(value))
