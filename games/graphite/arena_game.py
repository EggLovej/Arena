"""Graphite as the Arena platform sees it: the adapter between this folder and ``platform/arena``.

Everything the platform needs from Graphite goes through ``GAME``: its maps and
bots, new matches, what to draw for each tick, and actions as JSON. The rules
themselves stay in ``game.py`` and the bots in ``ais/``, unchanged.

Graphite's modules import each other by their plain names (``from game import
Game``), so loading this adapter puts this folder first on ``sys.path``. That
is also why one process plays one game's matches.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Leaderboard matches play the networks committed in checkpoints/, not whatever a
# local training run last wrote to ml/runs/. Setting these beforehand plays others.
os.environ.setdefault("NEURAL_AI_CHECKPOINT", str(ROOT / "checkpoints" / "neural_ai.pt"))
os.environ.setdefault("BEAST_AI_FILE", str(ROOT / "checkpoints" / "beast.json"))

from game import Game as Engine  # noqa: E402
from game_types import BLUE, GREEN, NEUTRAL, PLAYERS, RED, SendAction  # noqa: E402
from bot_names import bot_name  # noqa: E402
from human_controller import SEND_FRACTION, HumanController  # noqa: E402
from loading import available_ais, available_maps, load_map  # noqa: E402
from sizing import LARGE_MAP  # noqa: E402

from arena.contract import BotInfo, GameInfo, MapInfo, Outcome, SeatInfo, bot_from_module, summary  # noqa: E402


SEAT_OF = {RED: "red", BLUE: "blue", GREEN: "green"}
PLAYER_OF = {seat: player for player, seat in SEAT_OF.items()}
THEME_FILE = ROOT / "themes" / "paper_atlas.json"
THEME = json.loads(THEME_FILE.read_text())

# Bots in ais/ that the Arena leaves out. The original Atlas forecasts each opponent
# by running a copy of it (the scripted AIs' code, or Instinct's network), so it
# thinks for minutes per match; Atlas Clean plays its search without those copies.
HIDDEN = {"atlas_ai"}
# Bots that use a neural network.
NEURAL = {"neural_ai", "beast_ai"}
# Map groups a bot leaves out of tournaments. On the large maps Atlas Clean's search
# takes over half a second a move, and the networks spend over 15 minutes on one match
# (they were trained on maps of at most 30 bases and three-minute matches).
LARGE_MAPS = "Large maps"
SITS_OUT = {name: (LARGE_MAPS,) for name in ("atlas_clean_ai", "neural_ai", "beast_ai")}
# What the website says about each bot. A bot not listed here is described by the
# first paragraph of its module's docstring.
BLURBS = {
    "aggressive_ai": "Attacks often, with whatever it has to hand.",
    "careful_ai": "Builds up and waits until an attack should succeed.",
    "strategic_ai": "Plans attacks, defence and reinforcement routes on any map. "
    "The leaderboard's fixed point: it is rated 1000.",
    "neural_ai": "A neural network trained on Graphite games. It picks each move from what it learned, "
    "without looking ahead.",
    "beast_ai": "Our strongest player so far, crowned by beating the previous one. Today that is a network "
    "that beat the last Beast in 61% of 200 games.",
    "atlas_clean_ai": "Atlas's look-ahead search without copies of its opponents: it forecasts each one "
    "with whichever of its own ten styles best explains the fleets it has seen.",
}
INFO = GameInfo(
    id="graphite",
    name="Graphite",
    tagline="Real-time strategy on a graph: grow your bases, send fleets, take the map.",
    seats=tuple(
        SeatInfo(seat, seat.title(), THEME["players"][seat]["fill"]) for seat in ("red", "blue", "green")
    ),
    tick_seconds=0.1,
    max_ticks=1800,  # three minutes of game time, as in the game's own evaluator
    rating_anchor=("strategic_ai", 1000.0),  # the scale of ais/ratings.json
)


class GraphiteMatch:
    """One match: Graphite's own ``Game`` plus what the platform needs on top."""

    def __init__(self, map_id: str, engine: Engine):
        self.map_id = map_id
        self.engine = engine
        self.seats = tuple(SEAT_OF[player] for player in engine.players)
        self.eliminated: dict[str, int] = {}  # seat -> tick it lost its last base and fleet
        # id() of an engine fleet -> (the fleet, its id in snapshots). Holding the fleet
        # keeps Python from reusing its id() for a new fleet while it is tracked.
        self._fleets: dict[int, tuple[object, str]] = {}
        self._fleet_departures: dict[str, int] = {}
        self._next_fleet = 0

    @property
    def tick(self) -> int:
        return self.engine.tick

    def acting(self) -> tuple[str, ...]:
        return self.seats

    def observe(self, seat: str):
        return self.engine.view_for(PLAYER_OF[seat])

    def step(self, actions: dict[str, object]) -> dict[str, str]:
        tick = self.engine.tick
        refused = self.engine.step({PLAYER_OF[seat]: action for seat, action in actions.items()})
        for fleet in self.engine.fleets:
            if id(fleet) not in self._fleets:
                fleet_id = str(self._next_fleet)
                self._next_fleet += 1
                self._fleets[id(fleet)] = (fleet, fleet_id)
                self._fleet_departures[fleet_id] = tick
        alive = {id(fleet) for fleet in self.engine.fleets}
        for key in [key for key in self._fleets if key not in alive]:
            self._fleet_departures.pop(self._fleets.pop(key)[1])
        living = set(self.engine.living_players())
        for seat in self.seats:
            if PLAYER_OF[seat] not in living and seat not in self.eliminated:
                self.eliminated[seat] = self.engine.tick
        return {SEAT_OF[player]: reason for player, reason in refused.items()}

    def finished(self) -> bool:
        return self.engine.is_finished()

    def outcome(self) -> Outcome:
        """Survivors first (sharing first place if time ran out), then the rest, last to fall first."""

        survivors = tuple(seat for seat in self.seats if seat not in self.eliminated)
        ranking = [survivors] if survivors else []
        for tick in sorted(set(self.eliminated.values()), reverse=True):
            ranking.append(tuple(seat for seat in self.seats if self.eliminated.get(seat) == tick))
        reason = "elimination" if self.finished() else "time"
        return Outcome(tuple(ranking), reason, self.engine.tick)

    def board(self) -> dict:
        definition = self.engine.map_definition
        return {
            "name": definition.name,
            "width": definition.width,
            "height": definition.height,
            "bases": [
                {"id": base.id, "x": base.x, "y": base.y, "production": base.production_rate}
                for base in definition.bases
            ],
            "edges": [{"a": edge.a, "b": edge.b, "time": edge.travel_time} for edge in definition.edges],
        }

    def snapshot(self) -> dict:
        """Each base's owner and whole units; each fleet as [owner, units, from, to, departure tick, travel time].

        A fleet's entry never changes while it travels: the website works out where
        it is from the tick, so a replay stores it once.
        """

        return {
            "bases": {base.id: [_seat(base.owner), int(base.units)] for base in self.engine.bases.values()},
            "fleets": {
                self._fleet_id(fleet): [
                    SEAT_OF[fleet.owner],
                    fleet.units,
                    fleet.source_id,
                    fleet.target_id,
                    self._fleet_departures[self._fleet_id(fleet)],
                    fleet.total_time,
                ]
                for fleet in self.engine.fleets
            },
        }

    def _fleet_id(self, fleet) -> str:
        return self._fleets[id(fleet)][1]

    def save_state(self) -> dict:
        return {
            "tick": self.engine.tick,
            "bases": {base.id: [base.owner, base.units] for base in self.engine.bases.values()},
            "fleets": [
                [
                    self._fleet_id(fleet),
                    self._fleet_departures[self._fleet_id(fleet)],
                    fleet.owner,
                    fleet.units,
                    fleet.source_id,
                    fleet.target_id,
                    fleet.remaining_time,
                    fleet.total_time,
                ]
                for fleet in self.engine.fleets
            ],
            "eliminated": dict(sorted(self.eliminated.items())),
            "next_fleet": self._next_fleet,
        }


class GraphiteGame:
    info = INFO

    def maps(self) -> tuple[MapInfo, ...]:
        maps = []
        for map_id in available_maps():
            definition = load_map(map_id)
            owners = {base.owner for base in definition.bases}
            seats = tuple(SEAT_OF[player] for player in PLAYERS if player in owners)
            description = f"{len(definition.bases)} bases, {len(definition.edges)} routes"
            if definition.tagline:
                description = f"{definition.tagline}. {description}"
            max_ticks = None
            if definition.time_limit is not None:
                max_ticks = round(definition.time_limit / INFO.tick_seconds)
                description += f", {definition.time_limit / 60:g} minutes"
            group = LARGE_MAPS if len(definition.bases) >= LARGE_MAP else ""
            maps.append(MapInfo(map_id, _title(definition.name), seats, description, max_ticks, group))
        return tuple(maps)

    def bots(self) -> tuple[BotInfo, ...]:
        return tuple(
            BotInfo(
                id=name,
                name=bot_name(name),
                description=BLURBS.get(name) or summary(ROOT / "ais" / f"{name}.py"),
                kind="neural" if name in NEURAL else "scripted",
                sits_out=SITS_OUT.get(name, ()),
            )
            for name in _roster()
        )

    def load_bot(self, bot_id: str):
        if bot_id not in _roster():
            raise ValueError(f"Graphite has no bot named {bot_id!r}")
        return bot_from_module(importlib.import_module(f"ais.{bot_id}"))

    def new_match(self, map_id: str, seed: int) -> GraphiteMatch:
        # Graphite's rules have no chance in them, so the seed only matters to bots.
        return GraphiteMatch(map_id, Engine(load_map(map_id), seconds_per_tick=INFO.tick_seconds))

    def action_to_json(self, action) -> list:
        if not isinstance(action, SendAction):
            raise TypeError(f"expected SendAction or None, got {type(action).__name__}")
        for name in ("source_id", "target_id"):
            if not isinstance(getattr(action, name), str):
                raise TypeError(f"{name} must be a base id (a string), got {type(getattr(action, name)).__name__}")
        if not isinstance(action.units, int) or isinstance(action.units, bool):
            raise TypeError(f"units must be a whole number, got {type(action.units).__name__}")
        return [action.source_id, action.target_id, action.units]

    def action_from_json(self, data: list) -> SendAction:
        source, target, units = data
        return SendAction(source, target, units)

    def person(self, seat: str) -> "WebPerson":
        return WebPerson(PLAYER_OF[seat])

    def web_needs(self, bot_id: str) -> dict:
        """The networks play through numpy in the browser, from their committed checkpoints."""

        if bot_id == "neural_ai":
            checkpoint = Path(os.environ["NEURAL_AI_CHECKPOINT"])
        elif bot_id == "beast_ai":
            player = json.loads(Path(os.environ["BEAST_AI_FILE"]).read_text())["player"]
            checkpoint = ROOT / player.removeprefix("network:").partition("?")[0]
        else:
            return {}
        return {"packages": ["numpy"], "files": [checkpoint.resolve().relative_to(ROOT).as_posix()]}

    def theme(self) -> dict:
        return THEME


class WebPerson(HumanController):
    """A person playing on the website, with the game window's controls (human_controller.py).

    The page sends what they did as commands:

    - ``{"send": [source, target]}``: half the source's units go to the target, as a click does
    - ``{"stream": [source, target]}``: from now on, every unit the source has goes there, as a shift-click does
    - ``{"stop": source}``: that base's stream ends

    Sends are played one a tick, in the order made, before any stream sends.
    """

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.sends: list[tuple[str, str]] = []

    def command(self, data: dict) -> None:
        if not isinstance(data, dict) or len(data) != 1:
            raise ValueError("expected one of send, stream or stop")
        (kind, value), = data.items()
        if kind == "stop" and isinstance(value, str):
            self.streams.pop(value, None)
        elif kind in ("send", "stream") and isinstance(value, list) and len(value) == 2 and all(isinstance(part, str) for part in value):
            if kind == "send":
                self.sends.append((value[0], value[1]))
            else:
                self.streams[value[0]] = value[1]
        else:
            raise ValueError(f"not a command: {data!r}")

    def act(self, view) -> SendAction | None:
        while self.sends:
            source_id, target_id = self.sends.pop(0)
            source = next((base for base in view.bases if base.id == source_id), None)
            if source is not None and source.owner == self.player_id and int(source.units) >= 1:
                return SendAction(source_id, target_id, max(1, int(int(source.units) * SEND_FRACTION)))
        return self.choose_action(view)

    def view(self) -> dict:
        # The streams from bases the person has lost are dropped by the next act().
        return {"streams": dict(self.streams)}


def _roster() -> tuple[str, ...]:
    return tuple(name for name in available_ais() if name not in HIDDEN)


def _seat(owner: int) -> str:
    return "" if owner == NEUTRAL else SEAT_OF[owner]


def _title(name: str) -> str:
    """"salt_and_pepper" -> "Salt and Pepper"."""

    small = {"and", "of", "the", "on", "in"}
    words = name.replace("_", " ").split()
    return " ".join(word if number and word in small else word.capitalize() for number, word in enumerate(words))


GAME = GraphiteGame()
