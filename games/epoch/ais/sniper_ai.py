"""Hides ranged units behind a short wall of melee ones: one melee unit for every three ranged,
a heavy unit at the front whenever it has twice its price, and in the Future every super soldier
it can pay for. It builds the age's cheapest turret once the enemy comes close and calls its
special when five enemy units are on the field.
"""

CLOSE = 250  # an enemy this near the base is a danger


def choose_action(observation):
    me = observation.me
    if observation.can("age"):
        return "age"
    if observation.can("special") and len(observation.enemy.units) >= 5:
        return "special"
    if observation.threat < CLOSE and observation.can("turret1"):
        return "turret1"
    if len(me.queue) >= 3:
        return None
    if observation.can("super"):
        return "super"
    if me.gold >= 2 * observation.price("heavy") and observation.can("heavy"):
        return "heavy"
    role = "melee" if 3 * me.count("melee") < me.count("ranged") else "ranged"
    return role if observation.can(role) else None
