"""A small example AI that attacks frequently."""

from game_types import SendAction


def choose_action(state):
    """Attack the weakest neighboring base from the strongest useful base."""

    choices = []
    for source in state.my_bases():
        available_units = int(source.units)
        if available_units < 2:
            continue

        for target in state.neighbors(source.id):
            if target.owner == state.player_id:
                continue
            choices.append((target.units, -source.units, source, target))

    if not choices:
        return None

    _, _, source, target = min(choices, key=lambda choice: choice[:2])
    units = max(1, int(source.units * 0.6))
    return SendAction(source.id, target.id, units)

