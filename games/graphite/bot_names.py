"""What each bot is called where people see it: the game's window and the Arena website.

Bots keep their file names (``strategic_ai``) as ids, so saved ratings, matches and
replays still point at the same bot after a rename. A bot not listed here is called
by its file name, for example ``my_new_ai`` -> ``My New``.
"""

NAMES = {
    "strategic_ai": "Strategic",
    "careful_ai": "Careful",
    "aggressive_ai": "Aggressive",
    "neural_ai": "Instinct",  # moves on what it learned, without looking ahead
    "beast_ai": "Apex",  # the strongest player so far, a title each new champion takes over
    "atlas_clean_ai": "Atlas",  # looks ahead, carrying the whole map on its shoulders
    "atlas_ai": "Mirror",  # the original Atlas, which forecasts opponents by running copies of them
}


def bot_name(name: str) -> str:
    """``strategic_ai`` (or ``strategic_ai.py``) -> ``Strategic``."""

    name = name.removesuffix(".py")
    return NAMES.get(name) or name.removesuffix("_ai").replace("_", " ").title()
