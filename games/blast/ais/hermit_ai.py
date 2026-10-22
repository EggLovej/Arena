"""Keeps its distance: it clears crates and picks up power-ups only where no rival is near, stays
out of places where a rival could trap it, and drops a bomb on a rival only when the blast would
leave that rival no way out. It means to be the last one standing rather than to win fights.
"""

from blast import Bot, distance

CROWD = 4  # tiles: a rival this near makes it keep away rather than dig


class Hermit(Bot):
    def choose(self, observation, plan):
        me = observation.me
        rivals = observation.rivals
        if any(plan.traps(rival) for rival in rivals if distance(me.tile, rival.tile) <= me.range + 3):
            return "bomb"
        crowded = any(distance(me.tile, rival.tile) <= CROWD for rival in rivals)
        target = None if crowded else plan.dig(self.chance)
        if target == me.tile and plan.worth(me.tile):
            return "bomb" if plan.can_bomb() else None
        if target is None:
            # The tile furthest from its nearest rival, of those it can stay on: once the walls
            # close in, of those they reach last.
            closes = observation.closes if plan.closing() else {}

            def room(tile):
                far = min((distance(tile, rival.tile) for rival in rivals), default=0)
                return closes.get(tile, 0), far, -plan.routes[tile].arrive

            target = max((tile for tile in plan.routes if plan.safe(tile)), key=room, default=None)
        action = plan.action(plan.routes.get(target))
        if action is None or plan.cornered(action):
            return None
        return action


def create_bot():
    return Hermit()
