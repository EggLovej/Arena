"""How a match of Blast plays out: bombers on a grid of walls and crates, tick by tick.

Each tick, in this order: bombs are dropped, then bombers step (so a bomb dropped
this tick stops a step onto its tile), each in seat order, the clock moves on, bombs whose fuse has
run out explode, flames that have burned out go, anyone standing in flames is out
of the match, and once the walls close in, the next tile of their spiral becomes
wall. A step moves a bomber onto the next tile at once and takes a few ticks
before it can step again; the website draws it walking across.

Everything a tick's blasts do is worked out from the field as it stood before
them, so the order bombs go off in never matters.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from blast.rules import (
    CLOSE_EVERY,
    CLOSING,
    CRATES,
    FLAME,
    FLOOR,
    FUSE,
    MAX_BOMBS,
    MAX_RANGE,
    MOVES,
    POWERS,
    SEATS,
    START_BOMBS,
    START_RANGE,
    STEPS,
    WALL,
    Layout,
)

Tile = tuple[int, int]
ACTIONS = (*MOVES, "bomb")


@dataclass
class Bomber:
    seat: str
    tile: Tile
    came_from: Tile  # where its last step began, for drawing
    stepped: int = -1  # the tick its last step began
    ready: int = 0  # the tick it can step again
    bombs: int = START_BOMBS  # how many it can have on the field at once
    range: int = START_RANGE
    speed: int = 0  # which of STEPS its steps take
    out: int = -1  # the tick it was caught, or -1 while it plays


@dataclass(frozen=True)
class Bomb:
    id: int
    tile: Tile
    owner: str
    explodes: int  # the tick it goes off, unless another blast sets it off sooner
    range: int


@dataclass(frozen=True)
class Seen:
    """A bomber as a bot sees it."""

    seat: str
    tile: Tile
    ready: int  # ticks until it can step again: 0 if it can now
    step: int  # ticks a step takes it
    bombs: int  # bombs it can still drop
    range: int


@dataclass(frozen=True)
class Observation:
    """What a bot sees: the whole field, except what the crates hide."""

    tick: int
    me: Seen
    rivals: tuple[Seen, ...]  # the other bombers still in the match
    walls: frozenset[Tile]  # the closed tiles included
    crates: frozenset[Tile]
    bombs: dict[Tile, Bomb]
    flames: dict[Tile, int]  # each burning tile and the tick its flames burn out
    powers: dict[Tile, str]  # power-ups lying in the open
    closes: dict[Tile, int]  # when each tile the walls have yet to reach becomes wall

    def open(self, tile: Tile) -> bool:
        """Whether the bot may step onto ``tile``: no wall, crate, bomb or other bomber there."""

        blocked = tile in self.walls or tile in self.crates or tile in self.bombs
        return not blocked and all(rival.tile != tile for rival in self.rivals)


class Field:
    def __init__(self, layout: Layout, seed: int):
        rows = layout.rows
        self.layout = layout
        self.walls = {(x, y) for y, row in enumerate(rows) for x, char in enumerate(row) if char == WALL}
        starts = {SEATS[int(char) - 1]: (x, y) for y, row in enumerate(rows) for x, char in enumerate(row) if char.isdigit()}
        self.bombers = {seat: Bomber(seat, starts[seat], starts[seat]) for seat in layout.seats}
        self.crates = deal(layout, seed)  # each crate and the power-up it hides, or None
        self.bombs: dict[Tile, Bomb] = {}
        self.flames: dict[Tile, list[int]] = {}  # each burning tile: [the tick it caught, the tick it burns out]
        self.powers: dict[Tile, str] = {}
        spiral = [tile for tile in _spiral(len(rows[0]), len(rows)) if tile not in self.walls]
        self.closes = {tile: CLOSING + number * CLOSE_EVERY for number, tile in enumerate(spiral)}
        self.closed: dict[Tile, int] = {}  # tiles the walls have closed, and when
        self.tick = 0
        self.next_bomb = 0

    def alive(self) -> list[Bomber]:
        return [bomber for bomber in self.bombers.values() if bomber.out < 0]

    def finished(self) -> bool:
        return len(self.alive()) <= 1

    def step(self, actions: dict[str, str | None]) -> dict[str, str]:
        """Play one tick; returns why any action was refused, by seat."""

        refused = {}
        # Bombs first, then steps, which a bomb dropped this tick may stand in the way of; each in
        # seat order, so of two bombers stepping onto one tile only the first gets there.
        acting = sorted((seat for seat in self.bombers if actions.get(seat) is not None), key=lambda seat: actions[seat] != "bomb")
        for seat in acting:
            bomber, action = self.bombers[seat], actions[seat]
            if reason := self.refusal(seat, action):
                refused[seat] = reason
            elif action == "bomb":
                self.bombs[bomber.tile] = Bomb(self.next_bomb, bomber.tile, seat, self.tick + FUSE, bomber.range)
                self.next_bomb += 1
            else:
                self._walk(bomber, action)

        self.tick += 1
        self._explode()
        self.flames = {tile: flame for tile, flame in self.flames.items() if flame[1] > self.tick}
        for bomber in self.alive():
            if bomber.tile in self.flames:
                bomber.out = self.tick
        self._close()
        return refused

    def refusal(self, seat: str, action: str) -> str | None:
        """Why the bomber in ``seat`` cannot take ``action`` now, or None if it can."""

        bomber = self.bombers[seat]
        if action not in ACTIONS:
            return f"{action!r} is not one of {', '.join(ACTIONS)}"
        if bomber.out >= 0:
            return "it is out of the match"
        if action == "bomb":
            if bomber.tile in self.bombs:
                return "a bomb is already there"
            if sum(bomb.owner == seat for bomb in self.bombs.values()) >= bomber.bombs:
                return f"all {bomber.bombs} of its bombs are out"
            return None
        if self.tick < bomber.ready:
            return "it is still stepping"
        x, y = bomber.tile
        dx, dy = MOVES[action]
        target = (x + dx, y + dy)
        for blocks, name in ((self.walls, "wall"), (self.crates, "crate"), (self.bombs, "bomb")):
            if target in blocks:
                return f"a {name} is in the way"
        if any(other.tile == target for other in self.alive()):
            return "another bomber is in the way"
        return None

    def _walk(self, bomber: Bomber, action: str) -> None:
        dx, dy = MOVES[action]
        bomber.came_from = bomber.tile
        bomber.tile = (bomber.tile[0] + dx, bomber.tile[1] + dy)
        power = self.powers.pop(bomber.tile, None)
        if power == "bomb":
            bomber.bombs = min(MAX_BOMBS, bomber.bombs + 1)
        elif power == "fire":
            bomber.range = min(MAX_RANGE, bomber.range + 1)
        elif power == "speed":
            bomber.speed = min(len(STEPS) - 1, bomber.speed + 1)
        bomber.stepped = self.tick
        bomber.ready = self.tick + STEPS[bomber.speed]

    def _explode(self) -> None:
        """Bombs whose fuse has run out go off, with every bomb their flames reach. Flames stop at
        walls, and at the first crate or bomb in their way, which they reach. Burned crates leave
        what they hid; power-ups that lay in the open burn."""

        blockers = self.crates.keys() | self.bombs.keys()
        going = [tile for tile, bomb in self.bombs.items() if bomb.explodes <= self.tick]
        burned = set()
        while going:
            bomb = self.bombs.pop(going.pop(), None)
            if bomb is None:
                continue  # it went off already, set off by another
            for tile in blast(bomb.tile, bomb.range, self.walls, blockers):
                burned.add(tile)
                if tile in self.bombs:
                    going.append(tile)
        for tile in burned:
            self.powers.pop(tile, None)
            self.flames[tile] = [self.tick, self.tick + FLAME]
        for tile in burned & self.crates.keys():
            if power := self.crates.pop(tile):
                self.powers[tile] = power

    def _close(self) -> None:
        """Once the walls close in, a tile of their spiral becomes wall every CLOSE_EVERY ticks,
        crushing whatever is there; a bomb there never goes off."""

        if self.tick < CLOSING or (self.tick - CLOSING) % CLOSE_EVERY:
            return
        tile = next((tile for tile, when in self.closes.items() if when == self.tick), None)
        if tile is None:
            return
        self.walls.add(tile)
        self.closed[tile] = self.tick
        for things in (self.crates, self.bombs, self.flames, self.powers):
            things.pop(tile, None)
        for bomber in self.alive():
            if bomber.tile == tile:
                bomber.out = self.tick

    def observe(self, seat: str) -> Observation:
        def seen(bomber: Bomber) -> Seen:
            left = bomber.bombs - sum(bomb.owner == bomber.seat for bomb in self.bombs.values())
            return Seen(bomber.seat, bomber.tile, max(0, bomber.ready - self.tick), STEPS[bomber.speed], left, bomber.range)

        return Observation(
            self.tick,
            seen(self.bombers[seat]),
            tuple(seen(bomber) for bomber in self.alive() if bomber.seat != seat),
            frozenset(self.walls),
            frozenset(self.crates),
            dict(self.bombs),
            {tile: flame[1] for tile, flame in self.flames.items()},
            dict(self.powers),
            {tile: when for tile, when in self.closes.items() if tile not in self.closed},
        )

    def snapshot(self) -> dict:
        """What the website draws. Each bomber: [x, y, the x and y its last step began at, the tick
        it began, the tick it can step again, bombs, range, speed, the tick it was caught or -1];
        each bomb: [x, y, owner, the tick it goes off]; each burning tile: [the tick it caught, the
        tick it burns out]; each closed tile: the tick it closed. Tiles are keyed "x,y"."""

        def key(tile: Tile) -> str:
            return f"{tile[0]},{tile[1]}"

        return {
            "bombers": {
                b.seat: [*b.tile, *b.came_from, b.stepped, b.ready, b.bombs, b.range, b.speed, b.out] for b in self.bombers.values()
            },
            "crates": {key(tile): 1 for tile in self.crates},
            "bombs": {str(bomb.id): [*bomb.tile, bomb.owner, bomb.explodes] for bomb in self.bombs.values()},
            "flames": {key(tile): flame for tile, flame in self.flames.items()},
            "powers": {key(tile): power for tile, power in self.powers.items()},
            "closed": {key(tile): when for tile, when in self.closed.items()},
        }

    def save_state(self) -> dict:
        return {
            "tick": self.tick,
            "next_bomb": self.next_bomb,
            "bombers": [[b.seat, *b.tile, *b.came_from, b.stepped, b.ready, b.bombs, b.range, b.speed, b.out] for b in self.bombers.values()],
            "crates": sorted([*tile, power] for tile, power in self.crates.items()),
            "bombs": [[bomb.id, *bomb.tile, bomb.owner, bomb.explodes, bomb.range] for bomb in self.bombs.values()],
            "flames": sorted([*tile, *flame] for tile, flame in self.flames.items()),
            "powers": sorted([*tile, power] for tile, power in self.powers.items()),
            "closed": sorted([*tile, when] for tile, when in self.closed.items()),
        }


def blast(origin: Tile, reach: int, walls, blockers) -> list[Tile]:
    """The tiles a bomb at ``origin`` burns: its own, and up to ``reach`` more in each direction,
    stopping before a wall and at the first of ``blockers`` (crates and bombs)."""

    tiles = [origin]
    for dx, dy in MOVES.values():
        x, y = origin
        for _ in range(reach):
            x, y = x + dx, y + dy
            if (x, y) in walls:
                break
            tiles.append((x, y))
            if (x, y) in blockers:
                break
    return tiles


def deal(layout: Layout, seed: int) -> dict[Tile, str | None]:
    """The crates a match starts with, and what each hides. Tiles that the layout's mirror maps
    onto each other share one draw, so starts that mirror each other have the same crates and
    power-ups around them."""

    chance = random.Random(seed)
    width, height = len(layout.rows[0]), len(layout.rows)
    draws: dict[Tile, str | None | bool] = {}  # False for no crate
    crates = {}
    for y, row in enumerate(layout.rows):
        for x, char in enumerate(row):
            if char != FLOOR:
                continue
            xs = {x, width - 1 - x} if "x" in layout.mirror else {x}
            ys = {y, height - 1 - y} if "y" in layout.mirror else {y}
            first = min((b, a) for a in xs for b in ys)
            if first not in draws:
                draws[first] = _draw(chance)
            if draws[first] is not False:
                crates[(x, y)] = draws[first]
    return crates


def _draw(chance: random.Random) -> str | None | bool:
    if chance.randrange(100) >= CRATES:
        return False
    pick = chance.randrange(100)
    for power, share in POWERS.items():
        if pick < share:
            return power
        pick -= share
    return None


def _spiral(width: int, height: int) -> list[Tile]:
    """Every tile inside the outer wall, ring by ring from the outside in, each ring clockwise from its top-left corner."""

    order = []
    left, top, right, bottom = 1, 1, width - 2, height - 2
    while left <= right and top <= bottom:
        order += [(x, top) for x in range(left, right + 1)]
        order += [(right, y) for y in range(top + 1, bottom + 1)]
        if top < bottom:
            order += [(x, bottom) for x in range(right - 1, left - 1, -1)]
        if left < right:
            order += [(left, y) for y in range(bottom - 1, top, -1)]
        left, top, right, bottom = left + 1, top + 1, right - 1, bottom - 1
    return order
