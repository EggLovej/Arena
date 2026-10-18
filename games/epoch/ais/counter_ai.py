"""Watches the enemy's army and trains what beats it: heavy units against melee, ranged against
heavy and melee against ranged. It builds a turret when pressed and saves up for each new age
once it has the experience for it.
"""

from epoch import COUNTERS

CLOSE = 250  # an enemy this near the base is a danger
BEATEN_BY = {weak: strong for strong, weak in COUNTERS.items()}


def choose_action(observation):
    me, enemy = observation.me, observation.enemy
    danger = observation.threat < CLOSE
    if observation.can("age"):
        return "age"
    if danger and not any(me.turrets) and observation.can("turret"):
        return "turret"
    if len(me.queue) >= 2:
        return None
    # The enemy's army by role, weighted by what it cost, with a melee front line to start.
    strength = {"melee": 1, "ranged": 0, "heavy": 0}
    for kind in (*(unit.kind for unit in enemy.units), *enemy.queue):
        strength[kind.role] += kind.price
    role = BEATEN_BY[max(strength, key=strength.get)]
    if role != "melee" and me.count("melee") == 0:
        role = "melee"  # something has to hold the front
    spare = me.gold if danger else observation.spare
    return role if observation.can(role) and observation.price(role) <= spare else None
