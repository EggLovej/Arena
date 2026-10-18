"""Epoch's own promises beyond what every game makes (test_games.py)."""

import pytest

from arena.live import PERSON, LiveMatch
from arena.registry import load_game

GAME = load_game("epoch")  # puts Epoch's folder on sys.path, for the imports below

from epoch import AGES, BASE_LENGTH, COUNTERS, PLACES, ROLES, Battle  # noqa: E402
from epoch.battle import Unit  # noqa: E402
from epoch.rules import BOUNTY, SPACING, START_GOLD, TICKS_PER_SECOND  # noqa: E402


def army(battle, side, kind, gold):
    """Puts as many of ``kind`` as ``gold`` buys onto the field, in their three files, 400 from the base."""

    for number in range(gold // kind.price):
        x = BASE_LENGTH + 400 - (number // 3) * (kind.length + SPACING) - kind.length // 2
        side.units.append(Unit(battle.next_id, kind, x, kind.hp, number % 3, (x, 0, False)))
        battle.next_id += 1


@pytest.mark.parametrize("age", range(len(AGES)), ids=[age.id for age in AGES])
@pytest.mark.parametrize("role", ROLES)
def test_each_role_beats_the_one_it_counters_for_the_same_gold(age, role):
    battle = Battle(2000, ("red", "blue"))
    units = dict(zip(ROLES, AGES[age].units))
    gold = 600 * (1, 3, 8)[age]
    army(battle, battle.sides[0], units[role], gold)
    army(battle, battle.sides[1], units[COUNTERS[role]], gold)
    while all(side.units for side in battle.sides) and battle.tick < 3000:
        battle.step({})
    assert battle.sides[0].units and not battle.sides[1].units


def test_an_undefended_base_falls_to_an_army():
    match = GAME.new_match("pass", 0)
    while not match.finished():
        match.step({"red": "melee", "blue": None})
    assert match.outcome().ranking == (("red",), ("blue",))
    assert match.outcome().reason == "elimination"


def test_when_time_runs_out_the_stronger_base_wins():
    match = GAME.new_match("meadow", 0)
    match.battle.side("blue").hp -= 1
    assert match.outcome().ranking == (("red",), ("blue",)) and match.outcome().reason == "time"
    match.battle.side("blue").hp += 1
    assert match.outcome().ranking == (("red", "blue"),)


def test_an_age_needs_experience_and_gold_and_strengthens_the_base():
    battle = Battle(1000, ("red", "blue"))
    red = battle.side("red")
    red.gold, red.xp = 500, 999
    assert battle.step({"red": "age"}) == {"red": "the Middle Ages needs 1000 XP"}
    red.gold, red.xp = 499, 1000
    assert battle.step({"red": "age"}) == {"red": "that costs 500 gold"}
    red.gold = 500
    assert battle.step({"red": "age"}) == {}
    assert (red.age, red.hp, red.gold) == (1, 900, 0)
    assert battle.observe("red").price("melee") == AGES[1].units[0].price


def test_turret_places_open_once_and_a_sold_turret_leaves_its_place_open():
    battle = Battle(1000, ("red", "blue"))
    red = battle.side("red")
    red.gold = 1000
    rock = AGES[0].turret.price
    # Build; open the second place and build; sell one and build again in its place; open the third.
    changes = (("turret", -rock), ("turret", -rock - PLACES[1]), ("sell", rock // 2), ("turret", -rock), ("turret", -rock - PLACES[2]))
    for action, change in changes:
        gold = red.gold
        assert battle.step({"red": action}) == {}
        assert red.gold == gold + change
    assert battle.step({"red": "turret"}) == {"red": "every turret place is taken"}
    assert [turret.kind.id for turret in red.turrets] == ["rock_thrower"] * 3


def test_kills_earn_gold_but_only_kills_by_units_earn_experience():
    clubman = AGES[0].units[0]
    match = GAME.new_match("pass", 0)
    red, blue = (match.battle.side(seat) for seat in ("red", "blue"))
    match.step({"red": "melee", "blue": "turret"})
    while red.queue or red.units:  # until the turret has killed the clubman
        match.step({})
    income = AGES[0].income * (match.tick // TICKS_PER_SECOND)
    assert blue.gold == START_GOLD - AGES[0].turret.price + income + clubman.price * BOUNTY // 100
    assert blue.xp == 0
    match = GAME.new_match("pass", 0)
    match.step({"red": "melee", "blue": "melee"})
    while any(side.queue or side.units for side in match.battle.sides):  # two clubmen kill each other
        match.step({})
    assert [side.xp for side in match.battle.sides] == [clubman.price, clubman.price]


def test_a_person_trains_units_from_the_browser():
    live = LiveMatch(GAME, "meadow", {"red": PERSON, "blue": "balanced_ai"}, seed=1)
    live.start()
    assert live.command("red", "fire")  # why it was refused
    for action in ("ranged", "melee", "age"):
        assert live.command("red", action) is None
    updates = [live.step() for _ in range(3)]  # one action a tick, in the order pressed
    assert live.match.snapshot()["queues"]["red"][1:] == ["slinger", "clubman"]
    assert updates[2]["refused"] == {"red": "the Middle Ages needs 1000 XP"}
