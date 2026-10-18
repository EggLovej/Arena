"""Every number in Epoch: the ages, their units and turrets, the economy and the lanes.

Balancing the game is a change to this file, and to the numbers ``RULES.md``
quotes. Distances are in board units (a lane is about a thousand long), times
in ticks of a tenth of a second, and every value is a whole number, so a match
plays out exactly the same anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

TICKS_PER_SECOND = 10
MAX_TICKS = 8 * 60 * TICKS_PER_SECOND  # eight minutes; then the base with more hit points wins

START_GOLD = 200
BOUNTY = 60  # percent of a unit's price its killer earns in gold; a kill units helped with also earns the price in XP
QUEUE = 5  # units waiting to leave a base, the one in training included
PLACES = (0, 150, 400)  # what it costs to open each of a base's turret places; the first is open from the start
REFUND = 50  # percent of its own price a sold turret returns
BASE_LENGTH = 120  # how much of the lane each base takes up
FILES = 3  # units march in this many files side by side, so three can fight at the front
SPACING = 2  # the gap a unit keeps behind the one in front of it in its file

# A unit or turret hits harder against the role it counters: melee against ranged,
# ranged against heavy, heavy against melee.
COUNTERS = {"melee": "ranged", "ranged": "heavy", "heavy": "melee"}
COUNTER_BONUS = 150  # percent of its damage
ROLES = ("melee", "ranged", "heavy")


@dataclass(frozen=True)
class UnitType:
    id: str
    name: str
    role: str  # "melee", "ranged" or "heavy"
    price: int
    train: int  # ticks in training
    hp: int
    damage: int
    cooldown: int  # ticks between attacks
    reach: int  # how far in front of it it hits
    speed: int  # board units a tick
    length: int  # how much of the lane it takes up


@dataclass(frozen=True)
class TurretType:
    id: str
    name: str
    price: int
    damage: int
    cooldown: int
    reach: int  # how far in front of the base it hits


@dataclass(frozen=True)
class Age:
    id: str
    name: str
    xp: int  # the experience a side needs to advance to this age
    price: int  # and the gold it costs
    income: int  # gold a side in this age gets every second
    base_hp: int  # the base's hit points in this age
    units: tuple[UnitType, UnitType, UnitType]  # melee, ranged, heavy
    turret: TurretType


AGES = (
    Age(
        "stone",
        "Stone Age",
        xp=0,
        price=0,
        income=8,
        base_hp=500,
        units=(
            UnitType("clubman", "Clubman", "melee", 15, 10, 60, 12, 10, 6, 3, 24),
            UnitType("slinger", "Slinger", "ranged", 28, 12, 40, 8, 12, 90, 3, 24),
            UnitType("mammoth_rider", "Mammoth rider", "heavy", 120, 30, 320, 30, 16, 10, 2, 72),
        ),
        turret=TurretType("rock_thrower", "Rock thrower", 120, 18, 15, 200),
    ),
    Age(
        "medieval",
        "Middle Ages",
        xp=1000,
        price=500,
        income=14,
        base_hp=900,
        units=(
            UnitType("knight", "Knight", "melee", 40, 12, 160, 31, 10, 6, 3, 24),
            UnitType("crossbowman", "Crossbowman", "ranged", 75, 14, 105, 21, 12, 110, 3, 24),
            UnitType("catapult", "Catapult", "heavy", 300, 36, 830, 78, 18, 16, 2, 72),
        ),
        turret=TurretType("arrow_tower", "Arrow tower", 300, 45, 12, 230),
    ),
    Age(
        "future",
        "Future",
        xp=3000,
        price=1500,
        income=24,
        base_hp=1500,
        units=(
            UnitType("blade_droid", "Blade droid", "melee", 100, 14, 420, 80, 10, 6, 4, 24),
            UnitType("laser_trooper", "Laser trooper", "ranged", 190, 16, 270, 55, 12, 110, 3, 24),
            UnitType("mech", "Mech", "heavy", 750, 40, 2150, 200, 18, 14, 2, 72),
        ),
        turret=TurretType("laser_turret", "Laser turret", 750, 120, 10, 260),
    ),
)

UNITS = {unit.id: unit for age in AGES for unit in age.units}
TURRETS = {age.turret.id: age.turret for age in AGES}
AGE_OF = {kind.id: number for number, age in enumerate(AGES) for kind in (*age.units, age.turret)}
# The march checks for enemies in reach before each step, so a step longer than a unit's reach
# could carry it past one.
assert all(kind.reach >= kind.speed for kind in UNITS.values())


@dataclass(frozen=True)
class Lane:
    """A map: one lane between the two bases."""

    id: str
    name: str
    length: int  # from one end to the other, bases included
    description: str


LANES = {
    lane.id: lane
    for lane in (
        Lane("meadow", "Meadow", 1000, "The usual lane."),
        Lane("pass", "Narrow pass", 700, "A short lane: armies meet fast and turrets cover more of it."),
        Lane("frontier", "Frontier", 1400, "A long lane: more time to build up before the armies meet."),
    )
}
