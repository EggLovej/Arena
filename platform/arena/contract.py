"""The contract every game and every bot follows.

Everything in ``arena`` (the match runner, replays, tournaments, ratings and the
data the website reads) talks to a game only through these types, so a new game
needs its rules, its bots and a small adapter, and nothing else here changes.

A game is a Python package in ``games/<id>/`` with an ``arena_game.py`` file
that defines ``GAME``, an object with the ``Game`` shape below.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


JSON = Any
Action = Any
Observation = Any


@dataclass(frozen=True)
class SeatInfo:
    """One place at the table, such as Red. ``color`` is how the website shows it."""

    id: str
    name: str
    color: str


@dataclass(frozen=True)
class GameInfo:
    id: str
    name: str
    tagline: str
    seats: tuple[SeatInfo, ...]  # every seat the game can have, in order
    tick_seconds: float  # game time per step
    max_ticks: int  # a match that reaches this many steps ends on time
    # A bot whose rating is pinned, for example ("strategic_ai", 1000). Without one the
    # average bot sits at 1500.
    rating_anchor: tuple[str, float] | None = None
    # How long a bot may think, in seconds: on one move (or loading and resetting), and in all
    # over one match. A bot that runs out of time forfeits the rest of the match.
    move_time_limit: float = 120.0
    match_time_limit: float = 1800.0


@dataclass(frozen=True)
class MapInfo:
    id: str
    name: str
    seats: tuple[str, ...]  # the seats this map uses
    description: str = ""
    max_ticks: int | None = None  # this map's own time limit, for maps that need longer than the game's
    group: str = ""  # a heading the website lists the map under, such as "Large maps"; "" for the main list


@dataclass(frozen=True)
class BotInfo:
    id: str
    name: str
    description: str = ""
    kind: str = "scripted"  # "scripted" or "neural"; the website shows it
    leaderboard: bool = True  # False keeps a bot out of tournaments
    seat_counts: tuple[int, ...] | None = None  # the match sizes it plays in tournaments; None for all


@dataclass(frozen=True)
class BotContext:
    """What a bot is told before a match starts."""

    game: str
    map: str
    seat: str
    seats: tuple[str, ...]
    seed: int
    tick_seconds: float
    max_ticks: int


@dataclass(frozen=True)
class Outcome:
    """How a match ended. ``ranking`` lists groups of seats, best first; a group shares its place."""

    ranking: tuple[tuple[str, ...], ...]
    reason: str  # for example "elimination" or "time"
    ticks: int

    @property
    def winner(self) -> str | None:
        return self.ranking[0][0] if self.ranking and len(self.ranking[0]) == 1 else None

    def place(self, seat: str) -> float:
        """1 for first; seats that share a place get the average of the places they share."""

        taken = 0
        for group in self.ranking:
            if seat in group:
                return taken + (len(group) + 1) / 2
            taken += len(group)
        raise KeyError(seat)

    def to_json(self) -> JSON:
        return {"ranking": [list(group) for group in self.ranking], "reason": self.reason, "ticks": self.ticks}

    @classmethod
    def from_json(cls, data: JSON) -> Outcome:
        return cls(tuple(tuple(group) for group in data["ranking"]), data["reason"], data["ticks"])


@runtime_checkable
class Bot(Protocol):
    """One player. A bot module defines ``create_bot()`` returning one of these."""

    def reset(self, context: BotContext) -> None: ...

    def act(self, observation: Observation) -> Action | None: ...


class Match(Protocol):
    """One running match. It owns the game state; the runner owns the clock and the bots."""

    seats: tuple[str, ...]
    tick: int

    def acting(self) -> tuple[str, ...]:
        """The seats that choose an action this tick: all of them in a simultaneous game."""

    def observe(self, seat: str) -> Observation: ...

    def step(self, actions: dict[str, Action | None]) -> dict[str, str]:
        """Advance one tick. ``actions`` has every seat in ``acting()``, with None for a seat that
        does nothing. Returns why any action was refused, by seat; refused actions do nothing."""

    def finished(self) -> bool: ...

    def outcome(self) -> Outcome:
        """The result as things stand; the runner asks for it when the match is finished or out of time."""

    def board(self) -> JSON:
        """What never changes during the match (the map), for drawing."""

    def snapshot(self) -> dict[str, dict[str, JSON]]:
        """What the website draws for this tick, as groups of keyed entries (see ``arena.frames``)."""

    def save_state(self) -> JSON:
        """The exact state, enough to continue the match from here."""


class Game(Protocol):
    info: GameInfo

    def maps(self) -> tuple[MapInfo, ...]: ...

    def bots(self) -> tuple[BotInfo, ...]: ...

    def load_bot(self, bot_id: str) -> Bot: ...

    def new_match(self, map_id: str, seed: int) -> Match: ...

    def restore_match(self, map_id: str, state: JSON) -> Match: ...

    def action_to_json(self, action: Action) -> JSON:
        """Raise ``TypeError`` or ``ValueError`` for something that is not an action. The match
        is given ``action_from_json`` of the result, so it plays exactly what the replay records."""

    def action_from_json(self, data: JSON) -> Action: ...

    def observation_to_json(self, observation: Observation) -> JSON: ...

    def rules(self) -> str:
        """The rules as Markdown, for the website."""

    def theme(self) -> JSON:
        """Colours and style the website's renderer uses."""

    def fingerprint(self) -> str:
        """Changes whenever the rules, maps or bots change (for example a hash of their files)."""


def state_hash(state: JSON) -> str:
    """A short, stable hash of a state, used to check that a replay plays back the same."""

    text = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class FunctionBot:
    """A bot that is a single function, ``choose_action(observation)``. It ignores ``reset``."""

    choose_action: Any
    contexts: list[BotContext] = field(default_factory=list)

    def reset(self, context: BotContext) -> None:
        self.contexts.append(context)

    def act(self, observation: Observation) -> Action | None:
        return self.choose_action(observation)


def bot_from_module(module) -> Bot:
    """``create_bot()`` if the module has one, else its ``choose_action`` wrapped as a bot."""

    create = getattr(module, "create_bot", None)
    if callable(create):
        bot = create()
        if not isinstance(bot, Bot):
            raise TypeError(f"{module.__name__}.create_bot() must return an object with reset() and act()")
        return bot
    choose_action = getattr(module, "choose_action", None)
    if callable(choose_action):
        return FunctionBot(choose_action)
    raise TypeError(f"{module.__name__} must define create_bot() or choose_action(observation)")
