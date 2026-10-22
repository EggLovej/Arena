"""Saves for heavy units, and in the Future for super soldiers, and sends each out with a melee
escort; otherwise it trains melee units only when the enemy comes close. It builds no turrets and
calls its special as soon as three enemy units are on the field.
"""

CLOSE = 300  # an enemy this near the base is a danger


def choose_action(observation):
    me = observation.me
    if observation.can("age"):
        return "age"
    if observation.can("special") and len(observation.enemy.units) >= 3:
        return "special"
    for role in ("super", "heavy"):
        if observation.can(role):
            return role
    if me.count("melee") < me.count("heavy") + me.count("super") or observation.threat < CLOSE:
        return "melee" if observation.can("melee") else None
    return None
