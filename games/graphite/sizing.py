"""Base sizes shared by the renderer and mouse hit-testing.

A base's radius grows with its production rate relative to the most productive
base on the same map, so the richest bases stand out on every map. Large maps
(LARGE_MAP bases or more) size every base by its own rate instead, so one
outsized prize doesn't shrink all the others.
"""

import math
from collections.abc import Iterable
from typing import Protocol


LARGE_MAP = 80


class _SizedBase(Protocol):
    id: str
    production_rate: float


def radius_range(base_count: int) -> tuple[int, int]:
    """Smallest and largest radius; crowded maps use smaller bases."""

    if base_count >= 25:
        return 17, 29
    if base_count >= 15:
        return 20, 34
    return 22, 40


def base_radii(bases: Iterable[_SizedBase]) -> dict[str, int]:
    bases = tuple(bases)
    if len(bases) >= LARGE_MAP:
        return {base.id: round(8 + 6 * math.sqrt(base.production_rate)) for base in bases}
    smallest, largest = radius_range(len(bases))
    max_rate = max((base.production_rate for base in bases), default=0.0)

    radii = {}
    for base in bases:
        share = base.production_rate / max_rate if max_rate > 0 else 0.5
        radii[base.id] = round(smallest + (largest - smallest) * share)
    return radii
