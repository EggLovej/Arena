"""The contract every game and every bot follows.

Everything in ``arena`` (the match runner, replays, tournaments, ratings and the
data the website reads) talks to a game only through these types, so a new game
needs its rules, its bots and a small adapter, and nothing else here changes.

A game is a folder ``games/<id>/`` with:

- ``arena_game.py``, which defines ``GAME``, an object with the ``Game`` shape below;
- ``RULES.md``, the rules as the website shows them;
- ``files.json``, the files its matches load besides ``arena_game.py``, as paths
  (a folder stands for every file in it) relative to the game's folder::

      {"rules": ["game.py", "maps"], "bots": ["ais", "checkpoints/net.pt"]}

  ``rules`` is what decides how a match plays out from the actions: the rules and
  maps. A replay depends on these and the adapter (``arena.registry.fingerprint``).
  ``bots`` is everything else a match loads: the bots, the code and data they
  use, and what the adapter imports. A tournament plays again when any of them
  changes, and the website's play page loads exactly these.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
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
    sits_out: tuple[str, ...] = ()  # map groups it leaves out of tournaments, such as ("Large maps",)


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
    """A game. Two more methods are optional, for the website's play page (``arena.live``):

    - ``person(seat)`` returns an ``arena.live.Person`` that turns a person's commands into actions.
      Without it, people can only watch the game's bots play.
    - ``web_needs(bot_id)`` names what the browser must load before that bot plays: Python
      ``packages`` (such as numpy) and data ``files`` from ``files.json`` (such as a network's
      weights), as ``{"packages": [...], "files": [...]}``.
    """

    info: GameInfo

    def maps(self) -> tuple[MapInfo, ...]: ...

    def bots(self) -> tuple[BotInfo, ...]: ...

    def load_bot(self, bot_id: str) -> Bot: ...

    def new_match(self, map_id: str, seed: int) -> Match: ...

    def action_to_json(self, action: Action) -> JSON:
        """Raise ``TypeError`` or ``ValueError`` for something that is not an action. The match
        is given ``action_from_json`` of the result, so it plays exactly what the replay records."""

    def action_from_json(self, data: JSON) -> Action: ...

    def theme(self) -> JSON:
        """Colours and style the website's renderer uses."""


def state_hash(state: JSON) -> str:
    """A short, stable hash of a state, used to check that a replay plays back the same."""

    text = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class FunctionBot:
    """A bot that is a single function, ``choose_action(observation)``. It ignores ``reset``."""

    choose_action: Any

    def reset(self, context: BotContext) -> None:
        pass

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


def summary(path: Path) -> str:
    """The first paragraph of a module's docstring, on one line, read without importing the module
    (so without its packages): what the website says about a bot that has no blurb of its own."""

    docstring = ast.get_docstring(ast.parse(path.read_text())) or ""
    return " ".join(docstring.split("\n\n")[0].split())
