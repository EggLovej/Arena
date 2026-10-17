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
A replay recorded before the game's rules or maps changed (``is_current``) may
no longer play back, and the next tournament replaces it.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

from arena.contract import Game, Outcome, state_hash
from arena.frames import apply
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


def is_current(game: Game, replay: dict) -> bool:
    """Whether the replay was recorded with the game's rules and maps as they are now."""

    return replay.get("fingerprint") in (None, game.fingerprint())


def verify(game: Game, replay: dict) -> list[str]:
    """Play the recorded actions again; returns what differs (an empty list if nothing)."""

    problems = []
    if replay["game"] != game.info.id:
        return [f"replay is for {replay['game']}, not {game.info.id}"]

    actions_by_tick: dict[int, dict[str, object]] = defaultdict(dict)
    for tick, seat, action in replay["actions"]:
        actions_by_tick[tick][seat] = game.action_from_json(action)

    match = game.new_match(replay["map"], replay["seed"])
    frames = replay["frames"]
    keyframe_every = frames["keyframe_every"]
    final_tick = frames["count"] - 1
    expected_keyframes = {str(tick) for tick in range(0, final_tick + 1, keyframe_every)}
    if len(frames["deltas"]) != final_tick or set(frames["keyframes"]) != expected_keyframes:
        return [f"the frames are not a keyframe every {keyframe_every} ticks and a delta for every tick"]

    previous: dict | None = None  # the replayed snapshot of the tick before
    differs = False

    def check(tick: int) -> None:
        # Every frame the website can show: each tick's delta from the one before, and each
        # keyframe. Only the first difference is reported, since a match that went another
        # way differs from then on.
        nonlocal previous, differs
        snapshot = _plain(match.snapshot())
        keyframe = frames["keyframes"].get(str(tick))
        if not differs and (
            (keyframe is not None and _plain(keyframe) != snapshot)
            or (previous is not None and _plain(apply(previous, frames["deltas"][tick - 1])) != snapshot)
        ):
            problems.append(f"snapshot differs at tick {tick}")
            differs = True
        previous = snapshot
        expected_hash = replay["hashes"].get(str(tick)) if tick % keyframe_every == 0 else None
        if expected_hash is not None and expected_hash != state_hash(match.save_state()):
            problems.append(f"state differs at tick {tick}")

    check(0)
    while match.tick < final_tick:
        # As in a live match: every acting seat, with None where its bot did nothing.
        match.step({seat: None for seat in match.acting()} | actions_by_tick.get(match.tick, {}))
        check(match.tick)

    if replay["hashes"].get("final") != state_hash(match.save_state()):
        problems.append("final state differs")
    if Outcome.from_json(replay["outcome"]) != match.outcome():
        problems.append(f"outcome differs: recorded {replay['outcome']}, replayed {match.outcome().to_json()}")
    return problems


def _plain(value):
    """The value as it reads back from JSON (tuples become lists)."""

    return json.loads(json.dumps(value))
