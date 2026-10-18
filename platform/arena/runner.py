"""Play one match between bots and record it."""

from __future__ import annotations

import hashlib
import random
import signal
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

from arena.contract import JSON, Action, BotContext, Game, Match, Outcome, state_hash
from arena.frames import FrameRecorder
from arena.registry import fingerprint


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
    first_error: str = ""

    def to_json(self) -> dict:
        return {
            "bot": self.bot,
            "moves": self.moves,
            "errors": self.errors,
            "refused": self.refused,
            "think_ms_mean": round(1000 * self.think_seconds / self.moves, 3) if self.moves else 0.0,
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


def max_ticks_for(game: Game, map_id: str) -> int:
    """How many ticks a match on this map may last: the map's own limit, or the game's."""

    return next((m.max_ticks for m in game.maps() if m.id == map_id and m.max_ticks), game.info.max_ticks)


class BotSeat:
    """One seat's bot, played the same way in tournaments and on the website: loaded and reset,
    then asked for a move on each tick it acts, timed, and its move read back as the replay
    records it. A bot that fails to load, or runs out of time, sits out the rest of the match;
    a bot that fails on one move loses that move. Time limits apply when ``timed``."""

    def __init__(self, game: Game, bot_id: str, context: BotContext, timed: bool = True):
        self.game = game
        self.report = SeatReport(bot_id)
        self.timed = timed
        limit = game.info.move_time_limit
        bot = None
        try:  # a broken bot loses its moves, not the whole tournament
            with self._limit(limit):
                bot = game.load_bot(bot_id)
                bot.reset(context)
        except OutOfTime:
            _note_error(self.report, f"{'load' if bot is None else 'reset'}: took longer than {limit:g}s")
            bot = None
        except Exception as error:  # a bot that fails to reset still plays; one that fails to load cannot
            _note_error(self.report, f"{'load' if bot is None else 'reset'}: {error!r}")
        self.bot = bot

    def move(self, match: Match, seat: str) -> tuple[Action | None, JSON]:
        """The bot's action for this tick and its JSON, or (None, None) if it has none."""

        report = self.report
        report.moves += 1
        if self.bot is None:
            return None, None
        observation = match.observe(seat)
        info = self.game.info
        limit = min(info.move_time_limit, info.match_time_limit - report.think_seconds)
        action = None
        before = time.perf_counter()
        try:
            with self._limit(limit):
                action = self.bot.act(observation)
        except OutOfTime:
            self.bot = None
            if limit < info.move_time_limit:
                _note_error(report, f"out of time: used up its {info.match_time_limit:g}s for the match")
            else:
                _note_error(report, f"out of time: thought for more than {limit:g}s on one move")
        except Exception as error:
            _note_error(report, repr(error))
        report.think_seconds += time.perf_counter() - before
        return checked(self.game, action, report)

    def _limit(self, seconds: float):
        return time_limit(seconds) if self.timed else nullcontext()


def checked(game: Game, action: Action | None, report: SeatReport) -> tuple[Action | None, JSON]:
    """The action as the game reads it back from its JSON, and that JSON; (None, None) for no action
    or for something that is not an action. The match plays the action as the replay records it, so
    a replay always plays back the same, whatever odd types a bot returned."""

    if action is None:
        return None, None
    try:
        encoded = game.action_to_json(action)
        return game.action_from_json(encoded), encoded
    except (TypeError, ValueError) as error:
        _note_error(report, f"not an action: {error}")
        return None, None


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
    max_ticks = max_ticks_for(game, map_id)
    if set(lineup) != set(match.seats):
        raise ValueError(f"Map {map_id} has seats {match.seats}, but the lineup names {tuple(lineup)}")

    seats = {
        seat: BotSeat(game, lineup[seat], BotContext(info.id, map_id, seat, match.seats, seat_seed(seed, seat), info.tick_seconds, max_ticks))
        for seat in match.seats
    }
    reports = {seat: bot_seat.report for seat, bot_seat in seats.items()}

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
    while not match.finished() and match.tick < max_ticks:
        tick = match.tick
        actions = {}
        for seat in match.acting():
            actions[seat], encoded = seats[seat].move(match, seat)
            if encoded is not None:
                actions_log.append([tick, seat, encoded])
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
            "fingerprint": fingerprint(info.id),
            "match_id": match_id,
            "map": map_id,
            "seed": seed,
            "tick_seconds": info.tick_seconds,
            "max_ticks": max_ticks,
            "seats": [{"seat": seat, "bot": lineup[seat]} for seat in match.seats],
            "board": match.board(),
            "frames": frames.to_json(),
            "actions": actions_log,
            "hashes": hashes,
            "outcome": outcome.to_json(),
        }
    return record


def _note_error(report: SeatReport, text: str) -> None:
    report.errors += 1
    if not report.first_error:
        report.first_error = text[:ERROR_TEXT_LIMIT]


class OutOfTime(BaseException):
    """A bot thought for too long. Not an ``Exception``, so a bot's own ``except Exception`` lets it through."""


@contextmanager
def time_limit(seconds: float):
    """Interrupt the code inside with ``OutOfTime`` after ``seconds``. This needs Unix signals and
    the main thread, which plays every match of the CLI and of the tournament's workers; anywhere
    else only a limit already used up (``seconds`` <= 0) applies."""

    if seconds <= 0:
        raise OutOfTime
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def on_alarm(signum, frame):
        raise OutOfTime

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
