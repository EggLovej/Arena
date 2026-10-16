"""A small example AI that waits until an attack should succeed."""

from game_types import NEUTRAL, SendAction


def choose_action(state):
    """Prefer valuable neutral bases, but only attack with an advantage."""

    choices = []
    for source in state.my_bases():
        available_units = int(source.units * 0.65)
        if available_units < 1:
            continue

        for target in state.neighbors(source.id):
            if target.owner == state.player_id:
                continue
            if available_units <= target.units + 1:
                continue

            neutral_bonus = 0 if target.owner == NEUTRAL else 1
            travel_time = state.travel_time(source.id, target.id)
            score = (
                neutral_bonus,
                -target.production_rate,
                target.units,
                travel_time,
            )
            choices.append((score, source, target, available_units))

    if not choices:
        return None

    _, source, target, units = min(choices, key=lambda choice: choice[0])
    return SendAction(source.id, target.id, units)

