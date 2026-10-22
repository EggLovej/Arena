"""Goes after the nearest rival from the start, picking up the power-ups it passes, and drops a
bomb whenever a rival stands in its blast's way. Crates in its path it blasts through.
"""

from blast import Bot, distance

NEAR = 3  # steps: a power-up this close is worth the detour


class Hunter(Bot):
    def choose(self, observation, plan):
        me = observation.me
        if not observation.rivals:
            return None
        hit = plan.blast_from(me.tile)
        if any(rival.tile in hit for rival in observation.rivals) and plan.can_bomb():
            return "bomb"
        for tile in observation.powers:
            route = plan.routes.get(tile)
            if route and plan.safe(tile) and route.arrive - observation.tick <= NEAR * me.step:
                return plan.action(route)
        prey = min(observation.rivals, key=lambda rival: distance(rival.tile, me.tile))
        target = plan.nearest(prey.tile)
        if target == me.tile and prey.tile != me.tile:
            # As near as it can get: crates shut the way on, so it blasts them.
            if plan.worth(me.tile) and plan.can_bomb():
                return "bomb"
            target = plan.dig(self.chance)
        return plan.action(plan.routes.get(target))


def create_bot():
    return Hunter()
