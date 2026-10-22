"""Lets its turrets do the fighting, as Age of War players do: it opens every turret place it can
pay for, fills them with the dearest turret it can afford and, after each new age, sells the
oldest for better ones. Behind them it trains ranged units with a few melee ones in front, and
super soldiers once it can pay for them. It calls its special when the enemy reaches its base.
"""

from epoch import AGES

TURRETS = ("turret3", "turret2", "turret1")  # dearest first
CLOSE = 200  # an enemy this near the base is a danger


def choose_action(observation):
    me = observation.me
    age = AGES[me.age]
    if observation.can("age"):
        return "age"
    if observation.can("special") and observation.threat < CLOSE and (age.special.damage or me.units):
        return "special"
    if None in me.turrets:
        best = next((turret for turret in TURRETS if observation.can(turret)), None)
        if best and (best != "turret1" or observation.threat < CLOSE):
            return best
    elif observation.can("place"):
        return "place"
    elif any(t.kind.price < age.turrets[1].price for t in me.turrets) and me.gold >= age.turrets[2].price:
        return "sell"
    if len(me.queue) >= 2 or (None in me.turrets and observation.threat >= CLOSE):
        return None  # saving for a turret while it can
    if observation.can("super"):
        return "super"
    role = "melee" if 3 * me.count("melee") < me.count("ranged") + 1 else "ranged"
    return role if observation.can(role) else None
