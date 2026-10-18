"""Trains a steady mix of all three kinds of unit, adds a turret in each age and advances to the
next age as soon as it can. The leaderboard's fixed point: it is rated 1000.

It keeps about as many ranged units as melee ones and one heavy unit for every eight others,
counting those still in its base. Once it has the experience for the next age it saves its gold
for it, unless the enemy is at the gates.
"""

from epoch import AGE_OF, AGES

MIX = {"melee": 4, "ranged": 4, "heavy": 1}
CLOSE = 250  # an enemy this near the base is a danger


def choose_action(observation):
    me = observation.me
    danger = observation.threat < CLOSE
    spare = me.gold if danger else observation.spare
    if observation.can("age"):
        return "age"
    turrets = [turret for turret in me.turrets if turret]
    if len(turrets) <= me.age and (danger or spare >= 2 * observation.price("turret")) and observation.can("turret"):
        return "turret"
    if None not in me.turrets and any(AGE_OF[turret.kind.id] < me.age for turret in turrets):
        if spare >= 2 * AGES[me.age].turret.price:
            return "sell"
    if len(me.queue) >= 3:
        return None
    role = most_needed(me)
    return role if observation.can(role) and observation.price(role) <= spare else None


def most_needed(side) -> str:
    """The role furthest below its share of the mix."""

    total = sum(side.count(role) for role in MIX) + 1
    return max(MIX, key=lambda role: MIX[role] / sum(MIX.values()) - side.count(role) / total)
