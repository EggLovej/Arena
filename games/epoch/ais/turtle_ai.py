"""Fills every turret place first, then trains ranged units behind a few melee ones. After each
new age it sells its old turrets, one at a time, to put up new ones, and once its turrets are
the newest it saves for the next age or adds heavy units.
"""

from epoch import AGE_OF, AGES


def choose_action(observation):
    me = observation.me
    if observation.can("turret"):
        return "turret"
    if observation.can("age"):
        return "age"
    old = [turret for turret in me.turrets if turret and AGE_OF[turret.kind.id] < me.age]
    if old:
        return "sell" if me.gold >= AGES[me.age].turret.price else None
    spare = observation.spare
    if len(me.turrets) == 3 and spare >= 2 * observation.price("heavy") and observation.can("heavy"):
        return "heavy"
    role = "melee" if 2 * me.count("melee") < me.count("ranged") else "ranged"
    return role if observation.can(role) and observation.price(role) <= spare else None
