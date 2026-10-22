"""Clears crates until it has a second bomb and a longer blast, then stalks the nearest rival. It
drops a bomb on a rival only where the blast would leave it no way out, and keeps out of places
where a rival could do the same to it. Once the walls close in, it makes for the middle.
"""

from blast import Bot, distance

ARMED = 2, 3  # the bombs and the blast reach it wants before it hunts
STALK = 4  # steps: how far it looks for a tile to trap its prey from


class Trapper(Bot):
    def choose(self, observation, plan):
        me = observation.me
        if any(plan.traps(rival) for rival in observation.rivals):
            return "bomb"
        target = plan.dig(self.chance)
        if (me.bombs >= ARMED[0] and me.range >= ARMED[1]) or target is None or plan.closing():
            if observation.rivals:
                prey = min(observation.rivals, key=lambda rival: distance(me.tile, rival.tile))
                target = ambush(observation, plan, prey)
            if plan.closing() and (target is None or distance(target, me.tile) > STALK):
                target = plan.inmost()
        elif target == me.tile and plan.worth(me.tile):
            return "bomb" if plan.can_bomb() else None
        action = plan.action(plan.routes.get(target))
        if action is None or plan.cornered(action):
            return None
        return action


def ambush(observation, plan, prey):
    """Where to lie in wait for ``prey``: the nearest tile, a few steps off at most, that a bomb
    could trap it from as things stand; else the nearest from which its blast would reach the
    prey (but not the prey's own tile, where the prey could run as well as it)."""

    lines = set(plan.blast_from(prey.tile)) - {prey.tile}
    near = observation.tick + STALK * observation.me.step
    tiles = sorted((route.arrive, tile) for tile, route in plan.routes.items() if tile in lines and plan.safe(tile))
    for arrive, tile in tiles:
        if arrive <= near and plan.traps(prey, tile):
            return tile
    return tiles[0][1] if tiles else plan.nearest(prey.tile)


def create_bot():
    return Trapper()
