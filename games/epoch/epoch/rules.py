"""Every number in Epoch: the ages, their units, turrets and specials, the economy and the lanes.

Epoch follows Age of War (Louissi, 2007): its five ages, its prices for units,
turrets and turret places, its kill rewards and its experience for each age.
Age of War never published hit points or damage, so those are Epoch's own,
tuned in tournaments. Balancing the game is a change to this file, and to the
numbers ``RULES.md`` quotes. Distances are in board units (a lane is about a
thousand long), times in ticks of a tenth of a second, and every value is a
whole number, so a match plays out exactly the same anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

TICKS_PER_SECOND = 10
MAX_TICKS = 20 * 60 * TICKS_PER_SECOND  # twenty minutes; then the base with more hit points wins, or the side with more XP

START_GOLD = 200
BOUNTY = 130  # percent of a unit's price its killer earns in gold
XP_KILL = 400  # percent of a unit's price its killer earns in experience
XP_LOSS = 50  # and its own side, for losing it
QUEUE = 5  # units waiting to leave a base, the one in training included
PLACES = (0, 1000, 3000, 7500)  # what it costs to open each of a base's turret places; the first is open from the start
REFUND = 50  # percent of its own price a sold turret returns
SPECIAL_COOLDOWN = 45 * TICKS_PER_SECOND  # ticks between a side's specials
BASE_LENGTH = 120  # how much of the lane each base takes up
SPACING = 2  # the gap a unit keeps behind the one in front of it
ROLES = ("melee", "ranged", "heavy", "super")


@dataclass(frozen=True)
class UnitType:
    id: str
    name: str
    role: str  # one of ROLES
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
class Special:
    id: str
    name: str
    damage: int  # to every enemy unit on the field; a special without damage heals every unit of its side instead


@dataclass(frozen=True)
class Age:
    id: str
    name: str
    xp: int  # the experience a side needs to advance to this age
    income: int  # gold a side in this age gets every second, besides what it earns from kills
    base_hp: int  # the base's hit points in this age
    units: tuple[UnitType, ...]  # melee, ranged and heavy, and in the last age a super unit
    turrets: tuple[TurretType, TurretType, TurretType]  # cheapest first
    special: Special


AGES = (
    Age(
        "stone",
        "Stone Age",
        xp=0,
        income=2,
        base_hp=500,
        units=(
            UnitType("clubman", "Clubman", "melee", 15, 10, 60, 12, 10, 6, 3, 24),
            UnitType("slinger", "Slinger", "ranged", 25, 12, 40, 6, 12, 100, 3, 24),
            UnitType("mammoth_rider", "Mammoth rider", "heavy", 100, 30, 350, 35, 16, 10, 2, 72),
        ),
        turrets=(
            TurretType("rock_thrower", "Rock thrower", 100, 14, 15, 200),
            TurretType("sling_nest", "Sling nest", 200, 6, 3, 180),
            TurretType("boulder_catapult", "Boulder catapult", 500, 50, 15, 260),
        ),
        special=Special("rock_rain", "Rock rain", 80),
    ),
    Age(
        "medieval",
        "Middle Ages",
        xp=4000,
        income=5,
        base_hp=1300,
        units=(
            UnitType("knight", "Knight", "melee", 50, 10, 156, 31, 10, 6, 3, 24),
            UnitType("crossbowman", "Crossbowman", "ranged", 75, 12, 99, 15, 12, 110, 3, 24),
            UnitType("catapult", "Catapult", "heavy", 500, 30, 1115, 111, 16, 10, 2, 72),
        ),
        turrets=(
            TurretType("arrow_tower", "Arrow tower", 500, 29, 15, 210),
            TurretType("ballista", "Ballista", 750, 11, 3, 190),
            TurretType("oil_cauldron", "Oil cauldron", 1000, 65, 15, 270),
        ),
        special=Special("arrow_storm", "Arrow storm", 208),
    ),
    Age(
        "renaissance",
        "Renaissance",
        xp=14000,
        income=20,
        base_hp=3500,
        units=(
            UnitType("duelist", "Duelist", "melee", 200, 10, 420, 84, 10, 6, 3, 24),
            UnitType("musketeer", "Musketeer", "ranged", 400, 12, 307, 46, 12, 120, 3, 24),
            UnitType("cannon", "Cannon", "heavy", 1000, 30, 2122, 212, 16, 10, 2, 72),
        ),
        turrets=(
            TurretType("small_cannon", "Small cannon", 1500, 60, 15, 220),
            TurretType("large_cannon", "Large cannon", 3000, 26, 3, 200),
            TurretType("mortar", "Mortar", 6000, 192, 15, 280),
        ),
        special=Special("healing", "Healing", 0),
    ),
    Age(
        "modern",
        "Modern Age",
        xp=45000,
        income=150,
        base_hp=12000,
        units=(
            UnitType("commando", "Commando", "melee", 1500, 10, 1440, 288, 10, 6, 3, 24),
            UnitType("gunner", "Gunner", "ranged", 2000, 12, 859, 129, 12, 130, 3, 24),
            UnitType("tank", "Tank", "heavy", 7000, 30, 7028, 703, 16, 10, 2, 72),
        ),
        turrets=(
            TurretType("machine_gun", "Machine gun", 7000, 152, 15, 230),
            TurretType("rocket_launcher", "Rocket launcher", 9000, 52, 3, 210),
            TurretType("twin_gun", "Twin gun", 14000, 343, 15, 290),
        ),
        special=Special("air_strike", "Air strike", 1920),
    ),
    Age(
        "future",
        "Future",
        xp=200000,
        income=500,
        base_hp=30000,
        units=(
            UnitType("blade_droid", "Blade droid", "melee", 5000, 10, 3600, 720, 10, 6, 4, 24),
            UnitType("laser_trooper", "Laser trooper", "ranged", 6000, 12, 2036, 305, 12, 140, 3, 24),
            UnitType("mech", "Mech", "heavy", 20000, 30, 16267, 1627, 16, 10, 2, 72),
            UnitType("super_soldier", "Super soldier", "super", 150000, 60, 78872, 7887, 10, 8, 3, 32),
        ),
        turrets=(
            TurretType("laser_turret", "Laser turret", 24000, 302, 15, 240),
            TurretType("plasma_cannon", "Plasma cannon", 40000, 118, 3, 220),
            TurretType("ion_ray", "Ion ray", 100000, 984, 15, 300),
        ),
        special=Special("orbital_laser", "Orbital laser", 4800),
    ),
)

UNITS = {unit.id: unit for age in AGES for unit in age.units}
TURRETS = {turret.id: turret for age in AGES for turret in age.turrets}
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
