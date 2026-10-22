"""Takes the most pieces it can once the next player has replied as well as they can."""

from splice.search import Searcher


def create_bot():
    return Searcher(2)
