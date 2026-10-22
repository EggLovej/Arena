"""Epoch's own promises beyond what every game makes (test_games.py)."""

from arena.live import PERSON, LiveMatch
from arena.registry import load_game

GAME = load_game("epoch")  # puts Epoch's folder on sys.path, for the imports below

from epoch import AGES, BASE_LENGTH, PLACES, Battle  # noqa: E402
from epoch.battle import Unit  # noqa: E402
from epoch.rules import BOUNTY, SPACING, START_GOLD, TICKS_PER_SECOND, XP_KILL, XP_LOSS  # noqa: E402


def army(battle, side, kind, count):
    """Puts ``count`` of ``kind`` onto the field in single file, the first 400 from the base."""

    for number in range(count):
        x = BASE_LENGTH + 400 - number * (kind.length + SPACING) - kind.length // 2
        side.units.append(Unit(battle.next_id, kind, x, kind.hp, (x, 0, False)))
        battle.next_id += 1


def test_an_undefended_base_falls_to_an_army():
    match = GAME.new_match("pass", 0)
    while not match.finished():
        match.step({"red": "melee", "blue": None})
    assert match.outcome().ranking == (("red",), ("blue",))
    assert match.outcome().reason == "elimination"


def test_when_time_runs_out_the_stronger_base_wins_and_then_more_experience():
    match = GAME.new_match("meadow", 0)
    red, blue = match.battle.sides
    blue.hp -= 1
    assert match.outcome().ranking == (("red",), ("blue",)) and match.outcome().reason == "time"
    blue.hp, blue.xp = red.hp, 1
    assert match.outcome().ranking == (("blue",), ("red",))
    blue.xp = 0
    assert match.outcome().ranking == (("red", "blue"),)


def test_an_age_takes_only_experience_and_strengthens_the_base():
    battle = Battle(1000, ("red", "blue"))
    red = battle.side("red")
    red.gold, red.xp = 0, AGES[1].xp - 1
    assert battle.step({"red": "age"}) == {"red": f"the Middle Ages needs {AGES[1].xp} XP"}
    red.xp += 1
    assert battle.step({"red": "age"}) == {}
    assert (red.age, red.hp, red.gold) == (1, AGES[1].base_hp, 0)
    assert battle.observe("red").price("melee") == AGES[1].units[0].price
    assert battle.step({"red": "super"}) == {"red": "the Middle Ages has no super unit"}


def test_turret_places_cost_extra_and_a_sold_turret_leaves_its_place_open():
    battle = Battle(1000, ("red", "blue"))
    red = battle.side("red")
    red.gold = 20000
    rock, catapult = (turret.price for turret in AGES[0].turrets[::2])
    # Build in the first place; open the second and build the dearest there; sell, which sells the
    # cheaper, and build again in its place; open the last two.
    changes = (
        ("turret1", -rock),
        ("turret3", 0),  # refused: no place is free
        ("place", -PLACES[1]),
        ("turret3", -catapult),
        ("sell", rock // 2),
        ("turret1", -rock),
        ("place", -PLACES[2]),
        ("place", -PLACES[3]),
    )
    for action, change in changes:
        gold = red.gold
        refused = battle.step({"red": action})
        assert red.gold == gold + change and bool(refused) == (change == 0)
    assert battle.step({"red": "place"}) == {"red": "every turret place is open"}
    assert [turret and turret.kind.id for turret in red.turrets] == ["rock_thrower", "boulder_catapult", None, None]


def test_a_kill_earns_its_killer_gold_and_both_sides_experience():
    clubman = AGES[0].units[0]
    match = GAME.new_match("pass", 0)
    red, blue = (match.battle.side(seat) for seat in ("red", "blue"))
    match.step({"red": "melee", "blue": "turret1"})
    while red.queue or red.units:  # until the turret has killed the clubman
        match.step({})
    income = AGES[0].income * (match.tick // TICKS_PER_SECOND)
    assert blue.gold == START_GOLD - AGES[0].turrets[0].price + income + clubman.price * BOUNTY // 100
    assert (blue.xp, red.xp) == (clubman.price * XP_KILL // 100, clubman.price * XP_LOSS // 100)


def test_a_special_strikes_every_enemy_unit_or_heals_and_then_recharges():
    battle = Battle(2000, ("red", "blue"))
    red, blue = battle.sides
    army(battle, blue, AGES[0].units[0], 3)  # clubmen, which the Stone Age's special kills
    assert battle.step({"red": "special"}) == {} and not blue.units
    assert battle.step({"red": "special"}) == {"red": f"the {AGES[0].special.name} is ready in 45 seconds"}
    red.age = blue.age = next(number for number, age in enumerate(AGES) if not age.special.damage)
    army(battle, red, AGES[0].units[0], 2)
    for unit in red.units:
        unit.hp = 1
    while battle.step({"red": "special"}):  # refused until it has recharged
        pass
    assert [unit.hp for unit in red.units] == [unit.kind.hp for unit in red.units]


def test_a_person_trains_units_from_the_browser():
    live = LiveMatch(GAME, "meadow", {"red": PERSON, "blue": "balanced_ai"}, seed=1)
    live.start()
    assert live.command("red", "fire")  # why it was refused
    for action in ("ranged", "melee", "age"):
        assert live.command("red", action) is None
    updates = [live.step() for _ in range(3)]  # one action a tick, in the order pressed
    assert live.match.snapshot()["queues"]["red"][1:] == ["slinger", "clubman"]
    assert updates[2]["refused"] == {"red": f"the Middle Ages needs {AGES[1].xp} XP"}
