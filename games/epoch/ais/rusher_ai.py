"""Floods the lane with its cheapest units, two melee for every ranged one, and never builds a
turret. It advances to the next age only when it has gold left over, and with gold piling up
it adds heavy units too.
"""


def choose_action(observation):
    me = observation.me
    if observation.can("age"):
        return "age"
    if me.gold >= 3 * observation.price("heavy") and observation.can("heavy"):
        return "heavy"
    role = "ranged" if 2 * me.count("ranged") < me.count("melee") else "melee"
    return role if observation.can(role) else None
