"""How a match of Epoch plays out: two bases at the ends of one lane, and the armies between them.

Each tick, in this order: both sides' actions (train a unit, advance the age,
build or sell a turret), income, training, the fight and then the march.
Everything that fights in a tick fights at once, from where things stood
before anyone moved, so neither side gains from being first.

Each side measures its units' positions from its own end of the lane, so the
rules read the same for both and a side's units always march toward larger
numbers. A unit's position is its middle. Units march in three files side by
side, each file in single file, so three can fight at the front.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from epoch.rules import (
    AGE_OF,
    AGES,
    BASE_LENGTH,
    BOUNTY,
    COUNTER_BONUS,
    COUNTERS,
    FILES,
    PLACES,
    QUEUE,
    REFUND,
    ROLES,
    SPACING,
    START_GOLD,
    TICKS_PER_SECOND,
    TurretType,
    UnitType,
)

ACTIONS = (*ROLES, "age", "turret", "sell")


@dataclass
class Unit:
    id: int
    kind: UnitType
    x: int
    hp: int
    file: int  # which of the lane's files it marches in
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
        return refusal(self.me, action) is None

    def price(self, action: str) -> int:
        return price(self.me, action)

    @property
    def spare(self) -> int:
        """Gold the viewer can spend without putting off its next age: all of it, until it has the
        experience for that age."""

        me = self.me
        if me.age + 1 < len(AGES) and me.xp >= AGES[me.age + 1].xp:
            return me.gold - AGES[me.age + 1].price
        return me.gold

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
        reason = refusal(side, action)
        if reason:
            return reason
        side.gold -= price(side, action)
        if action in ROLES:
            if not side.queue:
                side.started = self.tick
            side.queue.append(AGES[side.age].units[ROLES.index(action)])
        elif action == "age":
            side.age += 1
            side.hp += AGES[side.age].base_hp - AGES[side.age - 1].base_hp
        elif action == "turret":
            if None not in side.turrets:
                side.turrets.append(None)  # a new place, paid for in price()
            side.turrets[side.turrets.index(None)] = Turret(AGES[side.age].turret)
        else:
            slot = _to_sell(side)
            side.gold += side.turrets[slot].kind.price * REFUND // 100
            side.turrets[slot] = None
        return None

    def _train(self, side: Side) -> None:
        """The first unit in the queue leaves the base once trained, into the file with the most room,
        if that has room for it."""

        if not side.queue or self.tick - side.started < side.queue[0].train:
            return
        kind = side.queue[0]
        clear = [self.length] * FILES  # how far each file is clear from the base: to its last unit
        for unit in side.units:  # front first, so the last of each file comes last
            clear[unit.file] = unit.back - SPACING
        file = max(range(FILES), key=clear.__getitem__)
        if clear[file] < BASE_LENGTH + kind.length:
            return
        x = BASE_LENGTH + kind.length // 2
        side.units.append(Unit(self.next_id, kind, x, kind.hp, file, (x, self.tick, False)))
        self.next_id += 1
        side.queue.pop(0)
        side.started = self.tick

    def _fight(self) -> None:
        """Every unit strikes the enemy base if it is in reach, else the front unit of the enemy army
        if that is; turrets strike the front unit. All strike at once; the dead leave at the end."""

        hits: list[tuple[Side, Unit | None, int, bool]] = []  # whose, which unit (None for the base), damage, by a unit?
        for side in self.sides:
            enemy = self.enemy(side)
            front = enemy.units[0] if enemy.units else None
            # How far the near edges of the enemy's front unit and base are from this side's end of the lane.
            near = self.length - front.front if front else None
            base = self.length - BASE_LENGTH
            for unit in side.units:
                unit.cooldown = max(0, unit.cooldown - 1)
                if unit.cooldown:
                    continue
                if base - unit.front <= unit.kind.reach:
                    hits.append((enemy, None, unit.kind.damage, True))
                elif front and near - unit.front <= unit.kind.reach:
                    # A heavy unit strikes every enemy in its reach; the others strike the nearest.
                    targets = enemy.units if unit.kind.role == "heavy" else [front]
                    for target in targets:
                        if self.length - target.front - unit.front > unit.kind.reach:
                            break
                        hits.append((enemy, target, _damage(unit.kind.damage, unit.kind.role, target), True))
                else:
                    continue
                unit.cooldown, unit.struck = unit.kind.cooldown, self.tick
            for turret in side.turrets:
                if turret is None:
                    continue
                turret.cooldown = max(0, turret.cooldown - 1)
                if turret.cooldown == 0 and front and near - BASE_LENGTH <= turret.kind.reach:
                    hits.append((enemy, front, turret.kind.damage, False))
                    turret.cooldown, turret.fired = turret.kind.cooldown, self.tick
        fought: set[int] = set()  # units struck by units: killing them earns experience, as turret kills don't
        for victim, unit, damage, by_unit in hits:
            if unit is None:
                victim.hp -= damage
            else:
                unit.hp -= damage
                if by_unit:
                    fought.add(unit.id)
        for side in self.sides:
            killer = self.enemy(side)
            for unit in side.units:
                if unit.hp <= 0:
                    killer.gold += unit.kind.price * BOUNTY // 100
                    killer.xp += unit.kind.price if unit.id in fought else 0
            side.units = [unit for unit in side.units if unit.hp > 0]

    def _march(self) -> None:
        """Each unit walks on unless an enemy is in reach or a friend is in the way. Both armies go
        from where the other stood before this march, so neither moves first."""

        edges = {}
        for side in self.sides:
            enemy = self.enemy(side)
            edges[side.seat] = self.length - (enemy.units[0].front if enemy.units else BASE_LENGTH)
        for side in self.sides:
            ahead = {}  # the back of the friend in front, in each file
            for unit in side.units:
                room = edges[side.seat] - unit.kind.reach - unit.front
                if unit.file in ahead:
                    room = min(room, ahead[unit.file] - SPACING - unit.front)
                step = max(0, min(unit.kind.speed, room))
                unit.x += step
                ahead[unit.file] = unit.back
                x0, t0, walking = unit.anchor
                if unit.x != (x0 + unit.kind.speed * (self.tick - t0) if walking else x0):
                    unit.anchor = (unit.x, self.tick, step == unit.kind.speed)
            side.units.sort(key=lambda unit: -unit.front)

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
            tuple(
                SeenUnit(unit.kind, self.length - unit.x if mirrored else unit.x, unit.hp, unit.cooldown)
                for unit in side.units
            ),
        )

    def snapshot(self) -> dict:
        """What the website draws. Each side: [age, base hit points, gold, XP, open turret places];
        each queue: [the tick the first unit began training, unit ids...]; each turret: [turret id,
        tick it last fired]; each unit: [seat, unit id, file, position, tick it was there, walking
        (1 or 0), hit points, tick it last struck]. A walking unit is ``speed`` further on for each
        tick after the one recorded."""

        units = {}
        for side in self.sides:
            for unit in side.units:
                x0, t0, walking = unit.anchor
                units[str(unit.id)] = [side.seat, unit.kind.id, unit.file, x0, t0, int(walking), unit.hp, unit.struck]
        return {
            "sides": {side.seat: [side.age, side.hp, side.gold, side.xp, len(side.turrets)] for side in self.sides},
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
                    "units": [[u.id, u.kind.id, u.x, u.hp, u.file, *u.anchor, u.cooldown, u.struck] for u in side.units],
                }
                for side in self.sides
            ],
        }


def price(side: Side | SeenSide, action: str) -> int:
    """The gold ``action`` costs ``side`` now: a turret with no open place free also opens the next
    place. (A sold turret's refund is not counted.)"""

    age = AGES[side.age]
    if action in ROLES:
        return age.units[ROLES.index(action)].price
    if action == "age":
        return AGES[side.age + 1].price if side.age + 1 < len(AGES) else 0
    if action == "turret":
        opening = None not in side.turrets and len(side.turrets) < len(PLACES)
        return age.turret.price + (PLACES[len(side.turrets)] if opening else 0)
    return 0


def refusal(side: Side | SeenSide, action: str) -> str | None:
    """Why ``side`` cannot take ``action`` now, or None if it can. A bot may ask about its own side
    (``observation.me``) before choosing."""

    if action not in ACTIONS:
        return f"{action!r} is not an action"
    cost = price(side, action)
    if action in ROLES and len(side.queue) >= QUEUE:
        return f"{QUEUE} units are already waiting"
    if action == "age":
        if side.age + 1 == len(AGES):
            return "this is the last age"
        if side.xp < AGES[side.age + 1].xp:
            return f"the {AGES[side.age + 1].name} needs {AGES[side.age + 1].xp} XP"
    if action == "turret" and None not in side.turrets and len(side.turrets) == len(PLACES):
        return "every turret place is taken"
    if action == "sell" and not any(side.turrets):
        return "there is no turret to sell"
    if side.gold < cost:
        return f"that costs {cost} gold"
    return None


def _to_sell(side: Side) -> int:
    """The place of the turret ``sell`` sells: the oldest kind, and of those the last built. Its place stays open."""

    built = [slot for slot, turret in enumerate(side.turrets) if turret]
    return min(reversed(built), key=lambda slot: AGE_OF[side.turrets[slot].kind.id])


def _damage(damage: int, role: str, target: Unit | None) -> int:
    if target is not None and COUNTERS[role] == target.kind.role:
        return damage * COUNTER_BONUS // 100
    return damage
