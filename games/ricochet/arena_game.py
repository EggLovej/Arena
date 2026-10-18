"""Ricochet as the Arena platform sees it: the adapter between this folder and ``platform/arena``.

Everything the platform needs from Ricochet goes through ``GAME``: its maps and
bots, new matches, what to draw for each tick, and actions as JSON. The rules
stay in ``src/ricochet/game`` and the bots in ``ais/``, unchanged.

Ricochet is an installable package (``src/ricochet``), so loading this adapter
puts ``src/`` on ``sys.path``. Unlike Graphite it needs no particular working
directory: maps and bots are always read from this folder by path.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ricochet.game.config import RulesConfig  # noqa: E402
from ricochet.game.engine import GameEngine  # noqa: E402
from ricochet.game.mapgen import generate_map  # noqa: E402
from ricochet.game.maps import available_maps, load_map  # noqa: E402
from ricochet.game.types import PLAYERS, ControlAction, MapDefinition, MatchContext, Vec2, validate_action  # noqa: E402
from ricochet.runtime.loading import available_ais, controller_label, load_controller  # noqa: E402
from ricochet.themes import NEON_CIRCUIT  # noqa: E402

from arena.contract import BotContext, BotInfo, GameInfo, MapInfo, Outcome, SeatInfo, summary  # noqa: E402

RULES = RulesConfig()
SEATS = ("cyan", "coral")  # player 0 and player 1, in Neon Circuit's colours
SEAT_OF = dict(zip(PLAYERS, SEATS))
PLAYER_OF = {seat: player for player, seat in SEAT_OF.items()}

# Seeds of the generated arenas the Arena plays besides the drawn maps in maps/, chosen
# for their variety. Each is "generated-<seed>" here and "generated:<seed>" in the game.
GENERATED = (3, 8, 12, 18, 21)
# Bots in ais/ that the Arena leaves out: the template is where a new bot starts.
HIDDEN = {"template_ai"}
# What the website says about each bot. A bot not listed here is described by the
# first paragraph of its module's docstring.
BLURBS = {
    "bank_shot_ai": "Shoots straight when it can see you, and banks shots off the top and bottom walls "
    "when it can't. Raises its shield when a shot comes close.",
    "close_range_ai": "Finds a way around the walls to you, leads its shots and attacks from close up.",
    "direct_ai": "Circles you, fires straight at you and sidesteps or shields against incoming shots. "
    "The leaderboard's fixed point: it is rated 1000.",
    "random_ai": "Wanders and fires in random directions.",
    "reactive_defender_ai": "Never attacks: it stands still and raises its shield just before a shot hits.",
    "sentinel_ai": "Our strongest bot. Plans routes around walls, aims where you will be, also off one wall, "
    "and dodges by tracing every shot a moment ahead. It never uses its shield.",
}

def _number(value: float) -> float | int:
    """A float for JSON, written as an integer when it is one (0 rather than 0.0)."""

    value = float(value)
    return int(value) if value.is_integer() else value


def _hex(color: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in color)


INFO = GameInfo(
    id="ricochet",
    name="Ricochet",
    tagline="A tank duel where every shot may bounce once: aim around walls, dodge, shield, outlast.",
    seats=tuple(SeatInfo(seat, seat.title(), _hex(color)) for seat, color in zip(SEATS, NEON_CIRCUIT.players)),
    tick_seconds=RULES.dt,
    max_ticks=RULES.max_ticks,  # 90 seconds at 30 ticks a second
    rating_anchor=("direct_ai", 1000.0),
)


def _definitions() -> dict[str, MapDefinition]:
    maps = {map_id: load_map(map_id, ROOT / "maps", RULES) for map_id in available_maps(ROOT / "maps")}
    maps.update({f"generated-{seed}": generate_map(seed, rules=RULES) for seed in GENERATED})
    return maps


MAPS = _definitions()


class RicochetBot:
    """One of Ricochet's controllers, told about the match the way Ricochet's own runtime tells it."""

    def __init__(self, controller):
        self.controller = controller

    def reset(self, context: BotContext) -> None:
        player = PLAYER_OF[context.seat]
        self.controller.reset(
            MatchContext(
                map_id=MAPS[context.map].id,
                player_id=player,
                opponent_id=1 - player,
                seed=context.seed,
                ticks_per_second=RULES.ticks_per_second,
                max_ticks=RULES.max_ticks,
                rules=RULES,
            )
        )

    def act(self, observation):
        return self.controller.act(observation)


class RicochetMatch:
    """One match: Ricochet's own ``GameEngine`` plus what the platform needs on top."""

    def __init__(self, map_id: str, engine: GameEngine):
        self.map_id = map_id
        self.engine = engine
        self.seats = SEATS
        # Each shot as drawn: [seat, x, y, vx, vy, tick, bounced], where (x, y) is where it was
        # at that tick. A shot flies straight, so this changes only when it bounces and a
        # replay stores it twice at most; the website works out where it is from the tick.
        self.shots: dict[str, list] = {}
        self.last_hit = {seat: -1 for seat in SEATS}  # the tick each tank was last hit

    @property
    def tick(self) -> int:
        return self.engine.tick

    def acting(self) -> tuple[str, ...]:
        return self.seats

    def observe(self, seat: str):
        return self.engine.observation_for(PLAYER_OF[seat])

    def step(self, actions: dict[str, ControlAction | None]) -> dict[str, str]:
        health = {seat: self.engine.tanks[PLAYER_OF[seat]].health for seat in SEATS}
        self.engine.step({PLAYER_OF[seat]: action for seat, action in actions.items() if action is not None})
        for seat in SEATS:
            if self.engine.tanks[PLAYER_OF[seat]].health < health[seat]:
                self.last_hit[seat] = self.engine.tick
        shots = {}
        for projectile in self.engine.projectiles:
            key = str(projectile.id)
            shot = self.shots.get(key)
            if shot is None or shot[6] != int(projectile.bounced):
                velocity = projectile.velocity
                shot = [
                    SEAT_OF[projectile.owner_id],
                    round(projectile.position.x, 1),
                    round(projectile.position.y, 1),
                    round(velocity.x, 1),
                    round(velocity.y, 1),
                    self.engine.tick,
                    int(projectile.bounced),
                ]
            shots[key] = shot
        self.shots = shots
        return {}  # Ricochet refuses nothing: an action it can't afford does what energy allows

    def finished(self) -> bool:
        return self.engine.terminated

    def outcome(self) -> Outcome:
        """The tank left standing wins; both falling together, or the clock running out, is a draw."""

        engine = self.engine
        winner = engine.winner
        if winner is not None:
            return Outcome(((SEAT_OF[winner],), (SEAT_OF[1 - winner],)), "elimination", engine.tick)
        destroyed = all(engine.tanks[player].health <= 0 for player in PLAYERS)
        return Outcome((SEATS,), "elimination" if destroyed else "time", engine.tick)

    def board(self) -> dict:
        definition = self.engine.map_definition
        return {
            "name": MAPS[self.map_id].name,
            "width": definition.width,
            "height": definition.height,
            "walls": [[wall.x, wall.y, wall.width, wall.height] for wall in definition.walls],
            "tank_radius": RULES.tank_radius,
            "shot_radius": RULES.projectile_radius,
            "health": RULES.starting_health,
            "energy": RULES.max_energy,
        }

    def snapshot(self) -> dict:
        """Each tank as [x, y, aim in degrees] and [health, energy, shield up, tick last hit]; each shot
        as in ``self.shots``. Positions are rounded to a tenth of a pixel, which is plenty to draw."""

        tanks, status = {}, {}
        for seat in SEATS:
            tank = self.engine.tanks[PLAYER_OF[seat]]
            aim = round(math.degrees(math.atan2(tank.aim.y, tank.aim.x))) % 360
            tanks[seat] = [round(tank.position.x, 1), round(tank.position.y, 1), aim]
            status[seat] = [tank.health, math.floor(tank.energy), int(tank.shield_active), self.last_hit[seat]]
        return {"tanks": tanks, "status": status, "shots": dict(self.shots)}

    def save_state(self) -> dict:
        return {"engine": self.engine.capture_state(), "shots": self.shots, "last_hit": self.last_hit}


class WebPerson:
    """A person playing on the website, with the game window's controls (src/ricochet/display/human.py):
    keys to move, the pointer to aim, buttons to fire and raise the shield. The page sends what is
    held whenever it changes, and it stays held until the next command:

    ``{"move": [x, y], "target": [x, y] or None, "fire": bool, "shield": bool}``

    ``target`` is the point on the board under the pointer; the tank aims at it every tick, from
    wherever it has moved to, as in the window. None keeps the aim it has.
    """

    def __init__(self):
        self.held = {"move": Vec2(0.0, 0.0), "target": None, "fire": False, "shield": False}

    def command(self, data: dict) -> None:
        try:
            move = _vector(data["move"]).limited()
            target = None if data["target"] is None else _vector(data["target"])
            fire, shield = data["fire"], data["shield"]
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"not a command: {error!r}") from None
        if not isinstance(fire, bool) or not isinstance(shield, bool):
            raise ValueError("fire and shield are true or false")
        self.held = {"move": move, "target": target, "fire": fire, "shield": shield}

    def act(self, observation) -> ControlAction:
        target = self.held["target"]
        aim = target - observation.me.position if target is not None else Vec2(0.0, 0.0)
        return ControlAction(self.held["move"], aim, self.held["fire"], self.held["shield"])

    def view(self) -> dict:
        return {}


def _vector(data) -> Vec2:
    x, y = data
    if isinstance(x, bool) or isinstance(y, bool) or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in (x, y)):
        raise ValueError(f"{data!r} is not two numbers")
    return Vec2(float(x), float(y))


class RicochetGame:
    info = INFO

    def maps(self) -> tuple[MapInfo, ...]:
        return tuple(
            MapInfo(map_id, definition.name, SEATS, _describe(definition)) for map_id, definition in MAPS.items()
        )

    def bots(self) -> tuple[BotInfo, ...]:
        return tuple(
            BotInfo(id=name, name=controller_label(name), description=BLURBS.get(name) or summary(ROOT / "ais" / f"{name}.py"))
            for name in _roster()
        )

    def load_bot(self, bot_id: str) -> RicochetBot:
        if bot_id not in _roster():
            raise ValueError(f"Ricochet has no bot named {bot_id!r}")
        return RicochetBot(load_controller(bot_id, ROOT / "ais"))

    def new_match(self, map_id: str, seed: int) -> RicochetMatch:
        # Ricochet's rules have no chance in them, so the seed only matters to bots.
        return RicochetMatch(map_id, GameEngine(MAPS[map_id], RULES, seed, verification=False))

    def person(self, seat: str) -> WebPerson:
        return WebPerson()

    def action_to_json(self, action) -> list | int:
        """[move x, move y, aim x, aim y, fire + 2 * shield], or 0 for doing nothing. Every tick of
        every bot is in the replay, so this is as short as it can be while exact: Sentinel's play
        against a shield changes if its moves are rounded even to a thousandth."""

        try:
            action = validate_action(action)
            numbers = [_number(value) for value in (action.move.x, action.move.y, action.aim.x, action.aim.y)]
        except (AttributeError, TypeError) as error:
            raise TypeError(f"expected a ControlAction of Vec2s and booleans: {error}") from None
        flags = int(action.fire) + 2 * int(action.shield)
        return [*numbers, flags] if any(numbers) or flags else 0

    def action_from_json(self, data: list | int) -> ControlAction:
        if data == 0:
            return ControlAction()
        move_x, move_y, aim_x, aim_y, flags = data
        return ControlAction(
            Vec2(float(move_x), float(move_y)), Vec2(float(aim_x), float(aim_y)), bool(flags & 1), bool(flags & 2)
        )

    def theme(self) -> dict:
        theme = NEON_CIRCUIT
        return {
            "name": theme.name,
            **{
                name: _hex(getattr(theme, name))
                for name in ("background", "panel", "text", "muted", "accent", "energy", "inactive", "shield")
            },
            "players": dict(zip(SEATS, map(_hex, theme.players))),
        }


def _roster() -> tuple[str, ...]:
    return tuple(name for name in available_ais(ROOT / "ais") if name not in HIDDEN)


def _describe(definition: MapDefinition) -> str:
    walls = f"{len(definition.walls)} wall{'s' if len(definition.walls) != 1 else ''}"
    if definition.id.startswith("generated:"):
        return f"Generated from seed {definition.id.split(':')[1]}, {walls}"
    return walls


GAME = RicochetGame()
