"""What a bot needs to stay alive and get about: when each tile will burn, and where a bomber can
walk without being caught on the way; and ``Bot``, the shape most bots take.

``Plan(observation)`` works both out for the bot itself, and ``Plan(observation, rival)`` for a
rival. Times are ticks of the match, as ``observation.tick`` counts them; a bomber stands on a
tile from the tick after it steps there, and is caught if the tile burns while it stands there.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, replace

from blast.field import Bomb, Observation, Seen, Tile, blast
from blast.rules import FLAME, FUSE, MOVES

FOREVER = 10**9
SOON = 100  # ticks ahead that the closing walls count: further off, they don't matter yet


def distance(a: Tile, b: Tile) -> int:
    """Steps from one tile to another with nothing in the way."""

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class Route:
    arrive: int  # the first tick the bomber can stand on the tile
    first: str | None  # its first step on the way, or None for the tile it is on
    leave: int  # the tick it takes that first step


def burns(observation: Observation, extra: tuple[Bomb, ...] = ()) -> dict[Tile, tuple[int, int]]:
    """When each tile will burn, as (the first tick, the last): from the flames burning now, the
    bombs on the field and ``extra`` bombs, and the walls closing in soon (a closed tile burns for
    ever). A bomb goes off when its fuse runs out or when the first blast reaches it, whichever is
    sooner, and reaches past the crates earlier blasts burn."""

    windows: dict[Tile, tuple[int, int]] = {}

    def burn(tile: Tile, first: int, last: int) -> None:
        old = windows.get(tile)
        windows[tile] = (min(old[0], first), max(old[1], last)) if old else (first, last)

    tick = observation.tick
    for tile, out in observation.flames.items():
        if out - 1 > tick:
            burn(tile, tick + 1, out - 1)
    bombs = {bomb.tile: bomb for bomb in (*observation.bombs.values(), *extra)}
    when = {tile: bomb.explodes for tile, bomb in bombs.items()}
    cleared: dict[Tile, int] = {}  # crates blasts will burn, and when: later blasts reach past them
    queue = [(goes, tile) for tile, goes in when.items()]
    heapq.heapify(queue)
    while queue:
        goes, tile = heapq.heappop(queue)
        if goes > when[tile]:
            continue  # set off sooner by another blast
        crates = {crate for crate in observation.crates if cleared.get(crate, goes) >= goes}
        for hit in blast(tile, bombs[tile].range, observation.walls, crates | bombs.keys()):
            burn(hit, goes, goes + FLAME - 1)
            if hit in crates:
                cleared.setdefault(hit, goes)
            if hit in when and when[hit] > goes:
                when[hit] = goes
                heapq.heappush(queue, (goes, hit))
    for tile, closes in observation.closes.items():
        if closes <= tick + SOON:
            burn(tile, closes, FOREVER)
    return windows


class Plan:
    """Where ``bomber`` (the bot itself unless given) can get to without being caught, as things
    stand plus any ``extra`` bombs, how soon, and how long it could last there. A bomber that has
    just dropped a bomb steps no sooner than the next tick: ``Plan.after_bomb`` plans for that."""

    def __init__(self, observation: Observation, bomber: Seen | None = None, extra: tuple[Bomb, ...] = ()):
        self.observation = observation
        self.bomber = bomber or observation.me
        self.windows = burns(observation, extra)
        self.routes = self._search(extra)

    @classmethod
    def after_bomb(cls, observation: Observation, tile: Tile | None = None) -> Plan:
        """The plan if the bot drops a bomb now where it stands, or on ``tile`` as if it stood there."""

        me = observation.me
        tile = tile or me.tile
        bomb = Bomb(-1, tile, me.seat, observation.tick + FUSE, me.range)
        return cls(observation, replace(me, tile=tile, ready=max(me.ready, 1)), (bomb,))

    def lasts(self, tile: Tile) -> int:
        """The last tick the bomber could stand on ``tile`` once there: FOREVER if nothing known
        burns it from then on, and -1 if it can't get there."""

        route = self.routes.get(tile)
        if route is None:
            return -1
        window = self.windows.get(tile)
        return FOREVER if window is None or window[1] < route.arrive else window[0] - 1

    def safe(self, tile: Tile) -> bool:
        """Whether the bomber can get to ``tile`` and stay there for good, as far as it knows."""

        return self.lasts(tile) == FOREVER

    def refuge(self) -> Tile:
        """The tile the bomber can last longest on, the soonest reached of those: one it is safe on
        if there is one."""

        return max(self.routes, key=lambda tile: (self.lasts(tile), -self.routes[tile].arrive))

    def escape(self) -> Route:
        """The way to ``refuge()``."""

        return self.routes[self.refuge()]

    def longest(self) -> int:
        """How long the bomber could last, at its refuge: FOREVER if it can get to safety."""

        return self.lasts(self.refuge())

    def can_bomb(self, tile: Tile | None = None) -> bool:
        """Whether the bomber has a bomb to drop where it stands (or on ``tile``) and could get to
        safety from it."""

        observation = self.observation
        tile = tile or self.bomber.tile
        if not self.bomber.bombs or tile in observation.bombs:
            return False
        return Plan.after_bomb(observation, tile).longest() == FOREVER

    def blast_from(self, tile: Tile) -> list[Tile]:
        """The tiles a bomb the bomber dropped on ``tile`` would burn, as the field stands."""

        observation = self.observation
        return blast(tile, self.bomber.range, observation.walls, observation.crates | observation.bombs.keys())

    def worth(self, tile: Tile) -> int:
        """The crates a bomb the bomber dropped on ``tile`` would break, leaving out those another blast is about to."""

        return sum(spot in self.observation.crates and spot not in self.windows for spot in self.blast_from(tile))

    def dig(self, chance: random.Random) -> Tile | None:
        """The best tile to go to for crates and power-ups, for the time the walk there takes (a
        power-up is worth three crates, and ``chance`` breaks ties): a tile to bomb crates from only
        if the bomber could get away from its own bomb there. None if nothing is worth a walk."""

        observation, bomber = self.observation, self.bomber
        scored = []
        for tile, route in self.routes.items():
            if not self.safe(tile):
                continue
            power = 3 * (tile in observation.powers)
            crates = self.worth(tile) if bomber.bombs and tile not in observation.bombs else 0
            if power or crates:
                steps = (route.arrive - observation.tick) / bomber.step
                scored.append(((power + crates) / (1 + steps) + chance.random() * 1e-3, tile, crates))
        for _, tile, crates in sorted(scored, reverse=True):
            if not crates or self.can_bomb(tile):
                return tile
        return None

    def nearest(self, target: Tile) -> Tile | None:
        """The tile the bomber can stay on that is nearest ``target`` on the grid, the soonest reached of those."""

        tiles = [tile for tile in self.routes if self.safe(tile)]
        return min(tiles, key=lambda tile: (distance(tile, target), self.routes[tile].arrive), default=None)

    def closing(self) -> bool:
        """Whether the walls are closing in, or soon will."""

        return min(self.observation.closes.values(), default=0) <= self.observation.tick + SOON

    def inmost(self) -> Tile | None:
        """The tile the bomber can stay on that the closing walls reach last, the soonest reached of those."""

        closes = self.observation.closes
        tiles = [tile for tile in self.routes if self.safe(tile)]
        return max(tiles, key=lambda tile: (closes.get(tile, 0), -self.routes[tile].arrive), default=None)

    def traps(self, rival: Seen, tile: Tile | None = None) -> bool:
        """Whether a bomb the bomber dropped where it stands now, or on ``tile`` as soon as it could
        get there, would cut ``rival``'s time short, to less than the bomber itself would last: the
        rival would be out first."""

        observation, me = self.observation, self.bomber
        tile = tile or me.tile
        if not me.bombs or tile in observation.bombs or tile not in self.routes:
            return False
        dropped = max(observation.tick, self.routes[tile].arrive - 1)
        theirs = Plan(observation, rival, (Bomb(-1, tile, me.seat, dropped + FUSE, me.range),)).longest()
        if theirs == FOREVER:
            return False
        return theirs < Plan(observation, rival).longest() and theirs < Plan.after_bomb(observation, tile).longest()

    def cornered(self, action: str) -> bool:
        """Whether a rival could trap the bomber on the tile ``action`` steps it to: a bomb dropped
        where the rival stands would leave the bomber there less time than it has."""

        observation = self.observation
        (x, y), (dx, dy) = self.bomber.tile, MOVES[action]
        tile = (x + dx, y + dy)
        there = replace(self.bomber, tile=tile, ready=self.bomber.step)
        longest = Plan(observation, there).longest()
        for rival in observation.rivals:
            if rival.bombs and distance(rival.tile, tile) <= rival.range + 1:
                bomb = Bomb(-1, rival.tile, rival.seat, observation.tick + FUSE, rival.range)
                if Plan(observation, there, (bomb,)).longest() < longest:
                    return True
        return False

    def action(self, route: Route | None) -> str | None:
        """What to do now to follow ``route``: its first step, once it is time."""

        if route is None or route.first is None or route.leave > self.observation.tick:
            return None
        return route.first

    def _search(self, extra: tuple[Bomb, ...]) -> dict[Tile, Route]:
        """The earliest the bomber can stand on each tile it can get to, waiting where that helps,
        never standing on a tile while it burns."""

        observation, bomber = self.observation, self.bomber
        # Rivals move, but they block a way while they stand in it.
        others = {seen.tile for seen in (observation.me, *observation.rivals) if seen.seat != bomber.seat}
        blocked = observation.walls | observation.crates | observation.bombs.keys() | {bomb.tile for bomb in extra} | others
        start = bomber.tile
        routes = {start: Route(observation.tick + 1, None, observation.tick + bomber.ready)}
        queue = [(observation.tick + 1, start)]
        while queue:
            arrive, tile = heapq.heappop(queue)
            route = routes[tile]
            if arrive > route.arrive:
                continue
            first, last = self.windows.get(tile, (FOREVER, FOREVER))
            stay = first - 1 if arrive <= first else FOREVER  # the last tick it can leave
            soonest = route.leave if tile == start else arrive - 1 + bomber.step
            for direction, (dx, dy) in MOVES.items():
                next_tile = (tile[0] + dx, tile[1] + dy)
                if next_tile in blocked:
                    continue
                leave = soonest
                burning = self.windows.get(next_tile)
                if burning and leave + bomber.step >= burning[0] and leave < burning[1]:
                    leave = burning[1]  # wait until its flames are out
                if leave > stay or leave > observation.tick + SOON:
                    continue
                old = routes.get(next_tile)
                if old is None or leave + 1 < old.arrive:
                    routes[next_tile] = Route(leave + 1, direction if tile == start else route.first, leave if tile == start else route.leave)
                    heapq.heappush(queue, (leave + 1, next_tile))
        return routes


class Bot:
    """The shape most bots take: it thinks only when it can step, and heads for its refuge (see
    ``Plan``) while it could last longer there; otherwise it does what ``choose`` says. ``chance`` draws from the
    match's seed, so a seeded match plays the same again."""

    def reset(self, context) -> None:
        self.chance = random.Random(context.seed)

    def act(self, observation: Observation) -> str | None:
        if observation.me.ready:
            return None
        plan = Plan(observation)
        if plan.lasts(observation.me.tile) < plan.longest():
            return plan.action(plan.escape())
        return self.choose(observation, plan)

    def choose(self, observation: Observation, plan: Plan) -> str | None:
        raise NotImplementedError
