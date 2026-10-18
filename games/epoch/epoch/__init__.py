"""Epoch: two bases at the ends of one lane, armies marching between them, through the ages.

``rules`` holds every number; ``battle`` plays a match. A bot is given an
``Observation`` each tick and answers with one of ``ACTIONS`` or None.
"""

from epoch.battle import ACTIONS, Battle, Observation
from epoch.rules import AGE_OF, AGES, BASE_LENGTH, COUNTERS, PLACES, ROLES

__all__ = ["ACTIONS", "AGE_OF", "AGES", "BASE_LENGTH", "COUNTERS", "PLACES", "ROLES", "Battle", "Observation"]
