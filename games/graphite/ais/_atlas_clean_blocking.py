"""Atlas Clean planning all at once, as Atlas does: for comparisons only (it freezes the game window)."""

from ais.atlas_clean_ai import AtlasClean


def new_player() -> AtlasClean:
    return AtlasClean(work_per_tick=None)


_shared = new_player()


def choose_action(state):
    return _shared(state)
