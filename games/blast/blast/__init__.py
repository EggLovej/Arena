"""Blast: bombers on a grid of walls and crates, and the last one standing wins.

``rules`` holds every number and reads the maps; ``field`` plays a match. A bot is
given an ``Observation`` each tick and answers with one of ``ACTIONS`` or None.
``plan`` works out what bots need most: when each tile will burn, and how to get
somewhere without being caught.
"""

from blast.field import ACTIONS, Bomb, Field, Observation
from blast.plan import Bot, Plan, distance
from blast.rules import MOVES

__all__ = ["ACTIONS", "MOVES", "Bomb", "Bot", "Field", "Observation", "Plan", "distance"]
