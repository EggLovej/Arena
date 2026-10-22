"""Clears crates for their power-ups: it walks to wherever one bomb breaks the most crates for the
walk, drops it there and takes cover. It bombs a rival only when one stands in its blast's way,
and makes for the middle once the walls close in. The leaderboard's fixed point: it is rated 1000.
"""

from blast import Bot


class Digger(Bot):
    def choose(self, observation, plan):
        me = observation.me
        target = plan.dig(self.chance)
        hit = plan.blast_from(me.tile)
        wanted = any(rival.tile in hit for rival in observation.rivals) or (target == me.tile and plan.worth(me.tile))
        if wanted and plan.can_bomb():
            return "bomb"
        if plan.closing():
            target = plan.inmost()
        elif target is None and observation.rivals:
            target = plan.nearest(observation.rivals[0].tile)
        return plan.action(plan.routes.get(target))


def create_bot():
    return Digger()
