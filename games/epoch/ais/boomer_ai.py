"""Races through the ages. It fights only for the experience the next age needs, then saves every
coin for it, training more only when the enemy comes close. In the last age it fields the
strongest units it can.
"""

from epoch import AGES

CLOSE = 300  # an enemy this near the base is a danger


def choose_action(observation):
    me = observation.me
    if observation.can("age"):
        return "age"
    safe = observation.threat >= CLOSE
    last = me.age + 1 == len(AGES)
    if safe and (len(me.queue) >= 2 or (not last and me.xp >= AGES[me.age + 1].xp)):
        return None  # enough on the way, or saving for the next age
    if last and 3 * me.count("heavy") <= me.count("melee") + me.count("ranged"):
        role = "heavy"
    else:
        role = "ranged" if me.count("ranged") < me.count("melee") else "melee"
    return role if observation.can(role) else None
