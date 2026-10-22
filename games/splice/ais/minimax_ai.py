"""Searches four moves ahead with alpha-beta, its own and the next three whoever makes them,
assuming everyone else plays their best against it."""

from splice.search import Searcher


def create_bot():
    return Searcher(4)
