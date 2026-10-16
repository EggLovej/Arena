"""Play one match between bots and record it."""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field

from arena.contract import BotContext, Game, Outcome, state_hash
from arena.frames import FrameRecorder


REPLAY_FORMAT = "arena-replay"
REPLAY_VERSION = 1
ERROR_TEXT_LIMIT = 300


@dataclass
class SeatReport:
    """How one seat's bot behaved: its failures and how long it thought."""

    bot: str
    moves: int = 0
    errors: int = 0  # the bot raised, or returned something that is not an action
    refused: int = 0  # the game refused the action, for example too many units
    think_seconds: float = 0.0
    slowest_seconds: float = 0.0
    first_error: str = ""

    def to_json(self) -> dict:
        return {
            "bot": self.bot,
            "moves": self.moves,
            "errors": self.errors,
            "refused": self.refused,
            "think_ms_mean": round(1000 * self.think_seconds / self.moves, 3) if self.moves else 0.0,
            "think_ms_max": round(1000 * self.slowest_seconds, 3),
            "first_error": self.first_error,
        }


@dataclass
class MatchRecord:
    match_id: str
    game: str
    map: str
    seed: int
    lineup: dict[str, str]  # seat -> bot id
    outcome: Outcome
    seats: dict[str, SeatReport]
    wall_seconds: float
    replay: dict | None = field(default=None, repr=False)

    def summary(self) -> dict:
        """One row of a tournament's results (everything but the replay)."""

        return {
            "id": self.match_id,
            "map": self.map,
            "seed": self.seed,
            "lineup": self.lineup,
            "outcome": self.outcome.to_json(),
            "seats": {seat: report.to_json() for seat, report in self.seats.items()},
            "wall_seconds": round(self.wall_seconds, 2),
        }


def seat_seed(seed: int, seat: str) -> int:
    digest = hashlib.sha256(f"{seed}:{seat}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def play_match(
    game: Game,
    map_id: str,
    lineup: dict[str, str],
    seed: int = 0,
    match_id: str | None = None,
    record_replay: bool = True,
    keyframe_every: int = 300,
) -> MatchRecord:
    """Play ``lineup`` (seat -> bot id) on one map until it ends or runs out of time."""

    info = game.info
    started = time.perf_counter()
    random.seed(seed)  # bots that use the random module play the same way again
    match = game.new_match(map_id, seed)
    if set(lineup) != set(match.seats):
        raise ValueError(f"Map {map_id} has seats {match.seats}, but the lineup names {tuple(lineup)}")

    reports = {seat: SeatReport(lineup[seat]) for seat in match.seats}
    bots = {}
    for seat in match.seats:
        bot = game.load_bot(lineup[seat])
        context = BotContext(info.id, map_id, seat, match.seats, seat_seed(seed, seat), info.tick_seconds, info.max_ticks)
        try:
            bot.reset(context)
        except Exception as error:  # a broken bot loses its moves, not the whole tournament
            _note_error(reports[seat], f"reset: {error!r}")
        bots[seat] = bot

    frames = FrameRecorder(keyframe_every) if record_replay else None
    actions_log: list[list] = []
    hashes: dict[str, str] = {}

    def observe_state() -> None:
        if frames is None:
            return
        frames.add(match.snapshot())
        if match.tick % keyframe_every == 0:
            hashes[str(match.tick)] = state_hash(match.save_state())

    observe_state()
    while not match.finished() and match.tick < info.max_ticks:
        tick = match.tick
        actions = {}
        for seat in match.acting():
            report = reports[seat]
            observation = match.observe(seat)
            before = time.perf_counter()
            try:
                action = bots[seat].act(observation)
            except Exception as error:
                action = None
                _note_error(report, repr(error))
            spent = time.perf_counter() - before
            report.moves += 1
            report.think_seconds += spent
            report.slowest_seconds = max(report.slowest_seconds, spent)
            if action is not None:
                try:
                    encoded = game.action_to_json(action)
                except (TypeError, ValueError) as error:
                    action = None
                    _note_error(report, f"not an action: {error}")
                else:
                    actions_log.append([tick, seat, encoded])
            actions[seat] = action
        for seat in match.step(actions):
            reports[seat].refused += 1
        observe_state()

    outcome = match.outcome()
    wall_seconds = time.perf_counter() - started
    match_id = match_id or f"{map_id}-{seed}"
    record = MatchRecord(match_id, info.id, map_id, seed, dict(lineup), outcome, reports, wall_seconds)
    if frames is not None:
        hashes["final"] = state_hash(match.save_state())
        record.replay = {
            "format": REPLAY_FORMAT,
            "version": REPLAY_VERSION,
            "game": info.id,
            "fingerprint": game.fingerprint(),
            "match_id": match_id,
            "map": map_id,
            "seed": seed,
            "tick_seconds": info.tick_seconds,
            "max_ticks": info.max_ticks,
            "seats": [{"seat": seat, "bot": lineup[seat]} for seat in match.seats],
            "board": match.board(),
            "frames": frames.to_json(),
            "actions": actions_log,
            "hashes": hashes,
            "outcome": outcome.to_json(),
            "reports": {seat: report.to_json() for seat, report in reports.items()},
        }
    return record


def _note_error(report: SeatReport, text: str) -> None:
    report.errors += 1
    if not report.first_error:
        report.first_error = text[:ERROR_TEXT_LIMIT]
