"""Trains a steady mix of melee and ranged units with now and then a heavy one, keeps a turret on
its base and advances to each new age as soon as it can. The leaderboard's fixed point: it is
rated 1000.

It keeps about as many ranged units as melee ones and one heavy unit for every eight others,
counting those still in its base, and in the Future trains a super soldier whenever it can pay
for one. With gold to spare it opens a second turret place, builds the age's middle turret in
each free place and sells the turrets of older ages for new ones. It calls its special when four
enemy units are on the field, or when its own army is hurt if the special heals.
"""

from epoch import AGES

MIX = {"melee": 4, "ranged": 4, "heavy": 1}
CLOSE = 250  # an enemy this near the base is a danger


def choose_action(observation):
    me, enemy = observation.me, observation.enemy
    age = AGES[me.age]
    if observation.can("age"):
        return "age"
    heals = not age.special.damage
    hurt = sum(unit.kind.hp - unit.hp for unit in me.units) > sum(unit.kind.hp for unit in me.units) // 3
    if observation.can("special") and (hurt if heals else len(enemy.units) >= 4):
        return "special"
    danger = observation.threat < CLOSE
    turret = observation.price("turret2")
    if None in me.turrets and (danger or me.gold >= 2 * turret) and observation.can("turret2"):
        return "turret2"
    if len(me.turrets) < 2 and me.gold >= 2 * observation.price("place") and observation.can("place"):
        return "place"
    if None not in me.turrets and any(t.kind.price < age.turrets[0].price for t in me.turrets) and me.gold >= 2 * turret:
        return "sell"
    if len(me.queue) >= 3:
        return None
    if observation.can("super"):
        return "super"
    role = most_needed(me)
    return role if observation.can(role) else None


def most_needed(side) -> str:
    """The role furthest below its share of the mix."""

    total = sum(side.count(role) for role in MIX) + 1
    return max(MIX, key=lambda role: MIX[role] / sum(MIX.values()) - side.count(role) / total)
