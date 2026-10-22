"""Floods the lane with its cheapest units, two melee for every ranged one, and never builds a
turret. It adds heavy units, and in the Future super soldiers, when gold piles up, advances as
soon as it can and calls its special whenever it is ready and two enemy units are on the field.
"""


def choose_action(observation):
    me = observation.me
    if observation.can("age"):
        return "age"
    if observation.can("special") and len(observation.enemy.units) >= 2:
        return "special"
    for role in ("super", "heavy"):
        if me.gold >= 3 * observation.price(role) and observation.can(role):
            return role
    role = "ranged" if 2 * me.count("ranged") < me.count("melee") else "melee"
    return role if observation.can(role) else None
