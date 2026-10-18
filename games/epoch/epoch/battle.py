"""How a match of Epoch plays out: two bases at the ends of one lane, and the armies between them.

Each tick, in this order: both sides' actions (train a unit, build, sell or make
room for a turret, advance the age, call the special), income, training, the
fight and then the march. Everything that fights in a tick fights at once, from
where things stood before anyone moved, so neither side gains from being first.

Each side measures its units' positions from its own end of the lane, so the
rules read the same for both and a side's units always march toward larger
numbers. A unit's position is its middle. Units march in single file, so only
the front one can reach an enemy close up; ranged ones behind it shoot past.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from epoch.rules import (
    AGES,
    BASE_LENGTH,
    BOUNTY,
    PLACES,
    QUEUE,
    REFUND,
    ROLES,
    SPACING,
    SPECIAL_COOLDOWN,
    START_GOLD,
    TICKS_PER_SECOND,
    XP_KILL,
    XP_LOSS,
    Age,
    TurretType,
    UnitType,
)

TURRET_ACTIONS = ("turret1", "turret2", "turret3")  # build the age's cheapest turret, its middle one or its dearest
ACTIONS = (*ROLES, *TURRET_ACTIONS, "place", "sell", "age", "special")


@dataclass
class Unit:
    id: int
    kind: UnitType
    x: int
    hp: int
    # Where it was and when, and whether it has walked on since: the website works out
    # where a walking unit is, so a snapshot changes only when a unit starts or stops.
    anchor: tuple[int, int, bool]
    cooldown: int = 0  # ticks until it can strike again
    struck: int = -1  # the tick it last struck, for drawing

    @property
    def front(self) -> int:
        return self.x + self.kind.length // 2

    @property
    def back(self) -> int:
        return self.x - self.kind.length // 2


@dataclass
class Turret:
    kind: TurretType
    cooldown: int = 0
    fired: int = -1


@dataclass
class Side:
    seat: str
    age: int = 0
    gold: int = START_GOLD
    xp: int = 0
    hp: int = AGES[0].base_hp
    queue: list[UnitType] = field(default_factory=list)
    started: int = 0  # the tick the first unit in the queue began training
    turrets: list[Turret | None] = field(default_factory=lambda: [None])  # one for each open place
    special: int = -SPECIAL_COOLDOWN  # the tick its special last struck
    units: list[Unit] = field(default_factory=list)  # front first


@dataclass(frozen=True)
class SeenUnit:
    kind: UnitType
    distance: int  # of its middle from the viewer's end of the lane
    hp: int
    cooldown: int


@dataclass(frozen=True)
class SeenSide:
    age: int
    gold: int
    xp: int
    hp: int
    queue: tuple[UnitType, ...]
    trained: int  # ticks the first unit in the queue has trained
    turrets: tuple[Turret | None, ...]
    special: int
    units: tuple[SeenUnit, ...]  # front first: for the enemy, the one nearest the viewer

    def count(self, role: str) -> int:
        """Units of this role on the field or waiting to leave the base."""

        return sum(unit.kind.role == role for unit in self.units) + sum(kind.role == role for kind in self.queue)


@dataclass(frozen=True)
class Observation:
    """What a bot sees: everything, from its own end of the lane. Nothing in Epoch is hidden."""

    tick: int
    lane: int  # the lane's length; a base's front is BASE_LENGTH from its end
    me: SeenSide
    enemy: SeenSide

    def can(self, action: str) -> bool:
        return refusal(self.me, action, self.tick) is None

    def price(self, action: str) -> int:
        return price(self.me, action)

    @property
    def threat(self) -> int:
        """How far the nearest enemy unit is from the front of the viewer's base."""

        nearest = self.enemy.units[0] if self.enemy.units else None
        edge = nearest.distance - nearest.kind.length // 2 if nearest else self.lane - BASE_LENGTH
        return edge - BASE_LENGTH


class Battle:
    def __init__(self, length: int, seats: tuple[str, str]):
        self.length = length
        self.sides = (Side(seats[0]), Side(seats[1]))
        self.tick = 0
        self.next_id = 0

    def side(self, seat: str) -> Side:
        return self.sides[0] if self.sides[0].seat == seat else self.sides[1]

    def enemy(self, side: Side) -> Side:
        return self.sides[1] if side is self.sides[0] else self.sides[0]

    def finished(self) -> bool:
        return any(side.hp <= 0 for side in self.sides)

    def step(self, actions: dict[str, str | None]) -> dict[str, str]:
        """Play one tick; returns why any action was refused, by seat."""

        refused = {}
        for side in self.sides:
            action = actions.get(side.seat)
            reason = self._act(side, action) if action is not None else None
            if reason:
                refused[side.seat] = reason
        self.tick += 1
        if self.tick % TICKS_PER_SECOND == 0:
            for side in self.sides:
                side.gold += AGES[side.age].income
        for side in self.sides:
            self._train(side)
        self._fight()
        self._march()
        return refused

    def _act(self, side: Side, action: str) -> str | None:
        reason = refusal(side, action, self.tick)
        if reason:
            return reason
        side.gold -= price(side, action)
        age = AGES[side.age]
        if action in ROLES:
            if not side.queue:
                side.started = self.tick
            side.queue.append(_unit(age, action))
        elif action in TURRET_ACTIONS:
            side.turrets[side.turrets.index(None)] = Turret(age.turrets[TURRET_ACTIONS.index(action)])
        elif action == "place":
            side.turrets.append(None)
        elif action == "sell":
            slot = _to_sell(side)
            side.gold += side.turrets[slot].kind.price * REFUND // 100
            side.turrets[slot] = None
        elif action == "age":
            side.age += 1
            side.hp += AGES[side.age].base_hp - age.base_hp
        else:
            side.special = self.tick + 1  # it strikes in this tick's fight; a healing one has healed already
            if not age.special.damage:
                for unit in side.units:
                    unit.hp = unit.kind.hp
        return None

    def _train(self, side: Side) -> None:
        """The first unit in the queue leaves the base once trained, if there is room behind the last
        unit out."""

        if not side.queue or self.tick - side.started < side.queue[0].train:
            return
        kind = side.queue[0]
        clear = side.units[-1].back - SPACING if side.units else self.length
        if clear < BASE_LENGTH + kind.length:
            return
        x = BASE_LENGTH + kind.length // 2
        side.units.append(Unit(self.next_id, kind, x, kind.hp, (x, self.tick, False)))
        self.next_id += 1
        side.queue.pop(0)
        side.started = self.tick

    def _fight(self) -> None:
        """A special strikes every enemy unit; every unit strikes the enemy base if it is in reach, else
        the front unit of the enemy army if that is; turrets strike the front unit. All strike at once;
        the dead leave at the end, earning their killers gold and both sides experience."""

        hits: list[tuple[Side, Unit | None, int]] = []  # whose, which unit (None for the base), damage
        for side in self.sides:
            enemy = self.enemy(side)
            front = enemy.units[0] if enemy.units else None
            # How far the near edges of the enemy's front unit and base are from this side's end of the lane.
            near = self.length - front.front if front else None
            base = self.length - BASE_LENGTH
            special = AGES[side.age].special
            if side.special == self.tick and special.damage:
                hits.extend((enemy, unit, special.damage) for unit in enemy.units)
            for unit in side.units:
                unit.cooldown = max(0, unit.cooldown - 1)
                if unit.cooldown:
                    continue
                if base - unit.front <= unit.kind.reach:
                    hits.append((enemy, None, unit.kind.damage))
                elif front and near - unit.front <= unit.kind.reach:
                    hits.append((enemy, front, unit.kind.damage))
                else:
                    continue
                unit.cooldown, unit.struck = unit.kind.cooldown, self.tick
            for turret in side.turrets:
                if turret is None:
                    continue
                turret.cooldown = max(0, turret.cooldown - 1)
                if turret.cooldown == 0 and front and near - BASE_LENGTH <= turret.kind.reach:
                    hits.append((enemy, front, turret.kind.damage))
                    turret.cooldown, turret.fired = turret.kind.cooldown, self.tick
        for victim, unit, damage in hits:
            if unit is None:
                victim.hp -= damage
            else:
                unit.hp -= damage
        for side in self.sides:
            killer = self.enemy(side)
            for unit in side.units:
                if unit.hp <= 0:
                    killer.gold += unit.kind.price * BOUNTY // 100
                    killer.xp += unit.kind.price * XP_KILL // 100
                    side.xp += unit.kind.price * XP_LOSS // 100
            side.units = [unit for unit in side.units if unit.hp > 0]

    def _march(self) -> None:
        """Each unit walks on unless an enemy is in reach or the friend in front is in the way. Both
        armies go from where the other stood before this march, so neither moves first."""

        edges = {}
        for side in self.sides:
            enemy = self.enemy(side)
            edges[side.seat] = self.length - (enemy.units[0].front if enemy.units else BASE_LENGTH)
        for side in self.sides:
            ahead = None  # the back of the friend in front
            for unit in side.units:
                room = edges[side.seat] - unit.kind.reach - unit.front
                if ahead is not None:
                    room = min(room, ahead - SPACING - unit.front)
                step = max(0, min(unit.kind.speed, room))
                unit.x += step
                ahead = unit.back
                x0, t0, walking = unit.anchor
                if unit.x != (x0 + unit.kind.speed * (self.tick - t0) if walking else x0):
                    unit.anchor = (unit.x, self.tick, step == unit.kind.speed)

    def observe(self, seat: str) -> Observation:
        side = self.side(seat)
        enemy = self.enemy(side)
        return Observation(self.tick, self.length, self._seen(side, False), self._seen(enemy, True))

    def _seen(self, side: Side, mirrored: bool) -> SeenSide:
        return SeenSide(
            side.age,
            side.gold,
            side.xp,
            side.hp,
            tuple(side.queue),
            self.tick - side.started if side.queue else 0,
            tuple(Turret(turret.kind, turret.cooldown) if turret else None for turret in side.turrets),
            side.special,
            tuple(
                SeenUnit(unit.kind, self.length - unit.x if mirrored else unit.x, unit.hp, unit.cooldown)
                for unit in side.units
            ),
        )

    def snapshot(self) -> dict:
        """What the website draws. Each side: [age, base hit points, gold, XP, open turret places,
        tick its special last struck]; each queue: [the tick the first unit began training, unit
        ids...]; each turret: [turret id, tick it last fired]; each unit: [seat, unit id, position,
        tick it was there, walking (1 or 0), hit points, tick it last struck]. A walking unit is
        ``speed`` further on for each tick after the one recorded."""

        units = {}
        for side in self.sides:
            for unit in side.units:
                x0, t0, walking = unit.anchor
                units[str(unit.id)] = [side.seat, unit.kind.id, x0, t0, int(walking), unit.hp, unit.struck]
        return {
            "sides": {
                side.seat: [side.age, side.hp, side.gold, side.xp, len(side.turrets), side.special] for side in self.sides
            },
            "queues": {side.seat: [side.started, *(kind.id for kind in side.queue)] for side in self.sides if side.queue},
            "turrets": {
                f"{side.seat}-{slot}": [turret.kind.id, turret.fired]
                for side in self.sides
                for slot, turret in enumerate(side.turrets)
                if turret
            },
            "units": units,
        }

    def save_state(self) -> dict:
        return {
            "tick": self.tick,
            "next_id": self.next_id,
            "sides": [
                {
                    "seat": side.seat,
                    "age": side.age,
                    "gold": side.gold,
                    "xp": side.xp,
                    "hp": side.hp,
                    "queue": [kind.id for kind in side.queue],
                    "started": side.started,
                    "turrets": [[t.kind.id, t.cooldown, t.fired] if t else None for t in side.turrets],
                    "special": side.special,
                    "units": [[u.id, u.kind.id, u.x, u.hp, *u.anchor, u.cooldown, u.struck] for u in side.units],
                }
                for side in self.sides
            ],
        }


def price(side: Side | SeenSide, action: str) -> int:
    """The gold ``action`` costs ``side`` now. (A sold turret's refund is not counted.)"""

    age = AGES[side.age]
    if action in ROLES:
        unit = _unit(age, action)
        return unit.price if unit else 0
    if action in TURRET_ACTIONS:
        return age.turrets[TURRET_ACTIONS.index(action)].price
    if action == "place":
        return PLACES[len(side.turrets)] if len(side.turrets) < len(PLACES) else 0
    return 0


def refusal(side: Side | SeenSide, action: str, tick: int) -> str | None:
    """Why ``side`` cannot take ``action`` at ``tick``, or None if it can. A bot may ask about its own
    side (``observation.can``) before choosing."""

    if action not in ACTIONS:
        return f"{action!r} is not an action"
    age = AGES[side.age]
    if action in ROLES:
        if _unit(age, action) is None:
            return f"the {age.name} has no {action} unit"
        if len(side.queue) >= QUEUE:
            return f"{QUEUE} units are already waiting"
    if action in TURRET_ACTIONS and None not in side.turrets:
        return "no turret place is free"
    if action == "place" and len(side.turrets) == len(PLACES):
        return "every turret place is open"
    if action == "sell" and not any(side.turrets):
        return "there is no turret to sell"
    if action == "age":
        if side.age + 1 == len(AGES):
            return "this is the last age"
        if side.xp < AGES[side.age + 1].xp:
            return f"the {AGES[side.age + 1].name} needs {AGES[side.age + 1].xp} XP"
    if action == "special":
        wait = side.special + SPECIAL_COOLDOWN - (tick + 1)  # ticks until a special called now could strike
        if wait > 0:
            return f"the {age.special.name} is ready in {math.ceil(wait / TICKS_PER_SECOND)} seconds"
    cost = price(side, action)
    if side.gold < cost:
        return f"that costs {cost} gold"
    return None


def _unit(age: Age, role: str) -> UnitType | None:
    return next((unit for unit in age.units if unit.role == role), None)


def _to_sell(side: Side) -> int:
    """The place of the turret ``sell`` sells: the cheapest, and of those the last built. Its place stays open."""

    built = [slot for slot, turret in enumerate(side.turrets) if turret]
    return min(reversed(built), key=lambda slot: side.turrets[slot].kind.price)
