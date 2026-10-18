"""Matches people play on the website, one tick at a time.

The website plays a match in the visitor's browser: Python runs there
(Pyodide), in a background worker (``web/public/workers/play.js``), and runs
this module. Each seat is a bot, which plays exactly as in a tournament, or a
person, whose clicks arrive as commands between ticks. The worker keeps the
clock: it asks every bot for its move as soon as a tick has been played
(``think``), waits until the tick is due, and plays it with whatever the
people chose meanwhile (``step``).

A game's optional ``person(seat)`` turns a person's commands into actions, and
``web_needs(bot_id)`` names what a bot needs loaded first (see ``arena.contract.Game``).
"""

from __future__ import annotations

import json
import random
from typing import Protocol

from arena.contract import JSON, Action, BotContext, Game, Observation
from arena.frames import delta
from arena.registry import load_game
from arena.runner import BotSeat, SeatReport, checked, max_ticks_for, seat_seed


PERSON = "person"  # stands in for a bot id where a person plays a seat


class Person(Protocol):
    """A seat a person plays from the website."""

    def command(self, data: JSON) -> None:
        """Something the person did, as the game's page sent it. Raise ``ValueError`` to refuse it."""

    def act(self, observation: Observation) -> Action | None:
        """This tick's action, from the commands so far."""

    def view(self) -> JSON:
        """What the page shows about this seat beyond the board, such as standing orders."""


def needs(game: Game, bot_ids) -> dict:
    """Everything the browser must load before these bots play: {"packages": [...], "files": [...]}."""

    packages, files = set(), set()
    for bot_id in bot_ids:
        wanted = getattr(game, "web_needs", lambda bot: {})(bot_id) or {}
        packages.update(wanted.get("packages", ()))
        files.update(wanted.get("files", ()))
    return {"packages": sorted(packages), "files": sorted(files)}


class LiveMatch:
    """One match on the website. ``lineup`` maps every seat to a bot id, or to ``PERSON``."""

    def __init__(self, game: Game, map_id: str, lineup: dict[str, str], seed: int):
        info = game.info
        self.game = game
        random.seed(seed)  # as in a tournament, for bots that use the random module
        self.match = game.new_match(map_id, seed)
        self.max_ticks = max_ticks_for(game, map_id)
        if set(lineup) != set(self.match.seats):
            raise ValueError(f"Map {map_id} has seats {self.match.seats}, but the lineup names {tuple(lineup)}")
        known = {bot.id for bot in game.bots()}
        self.people: dict[str, Person] = {}
        self.bots: dict[str, BotSeat] = {}
        self.reports: dict[str, SeatReport] = {}
        for seat in self.match.seats:
            if lineup[seat] == PERSON:
                if not hasattr(game, "person"):
                    raise ValueError(f"{info.name} cannot be played by people")
                self.people[seat] = game.person(seat)
                self.reports[seat] = SeatReport(PERSON)
                continue
            if lineup[seat] not in known:
                raise ValueError(f"{info.name} has no bot {lineup[seat]!r}")
            context = BotContext(info.id, map_id, seat, self.match.seats, seat_seed(seed, seat), info.tick_seconds, self.max_ticks)
            # As in a tournament, without the time limits: a slow bot slows the game down instead.
            self.bots[seat] = BotSeat(game, lineup[seat], context, timed=False)
            self.reports[seat] = self.bots[seat].report
        self._planned: dict[str, Action | None] | None = None  # the bots' actions for the coming tick
        self._shown = self.match.snapshot()

    def start(self) -> JSON:
        """What the page draws before the first tick."""

        return {"board": self.match.board(), "snapshot": self._shown, "tick": self.match.tick, **self._people_view()}

    def command(self, seat: str, data: JSON) -> str | None:
        """A person's command; returns why it was refused, or None."""

        if seat not in self.people:
            return f"{seat} is not played by a person"
        try:
            self.people[seat].command(data)
        except ValueError as error:
            return str(error)
        return None

    def think(self) -> None:
        """The bots acting in the coming tick choose their actions, from the position as it stands."""

        if self._planned is not None or self.finished():
            return
        self._planned = {seat: self.bots[seat].move(self.match, seat)[0] for seat in self.match.acting() if seat in self.bots}

    def step(self) -> JSON:
        """Play the coming tick with the bots' planned actions and the people's; returns what the page needs."""

        if self.finished():
            return self._update({})
        self.think()
        actions = dict(self._planned)
        self._planned = None
        problems = {}
        for seat in self.match.acting():  # a person whose turn it is not keeps their commands for later
            person = self.people.get(seat)
            if person is None:
                continue
            try:
                actions[seat] = checked(self.game, person.act(self.match.observe(seat)), self.reports[seat])[0]
            except Exception as error:  # a bug in the game's Person: the person loses this tick, not the match
                problems[seat] = repr(error)
                actions[seat] = None
        refused = self.match.step(actions)
        for seat in refused:
            self.reports[seat].refused += 1
        return self._update({seat: reason for seat, reason in {**refused, **problems}.items() if seat in self.people})

    def finished(self) -> bool:
        return self.match.finished() or self.match.tick >= self.max_ticks

    def _update(self, refused: dict[str, str]) -> JSON:
        snapshot = self.match.snapshot()
        changes = delta(self._shown, snapshot)
        self._shown = snapshot
        done = self.finished()
        return {
            "tick": self.match.tick,
            "delta": changes,
            "refused": refused,
            "errors": {seat: report.first_error for seat, report in self.reports.items() if report.first_error},
            "outcome": self.match.outcome().to_json() if done else None,
            **self._people_view(),
        }

    def _people_view(self) -> JSON:
        return {"people": {seat: person.view() for seat, person in self.people.items()}}


class Session:
    """The browser worker's hold on one game and its current match. Everything goes in and out
    as JSON text, so only strings cross between JavaScript and Python."""

    def __init__(self, game_id: str):
        self.game = load_game(game_id)
        self.match: LiveMatch | None = None

    def needs(self, lineup: str) -> str:
        return json.dumps(needs(self.game, [bot for bot in json.loads(lineup).values() if bot != PERSON]))

    def start(self, map_id: str, lineup: str, seed: int) -> str:
        self.match = LiveMatch(self.game, map_id, json.loads(lineup), int(seed))
        return json.dumps(self.match.start())

    def command(self, seat: str, data: str) -> str:
        return json.dumps(self.match.command(seat, json.loads(data)))

    def think(self) -> None:
        self.match.think()

    def step(self) -> str:
        return json.dumps(self.match.step())
