"""Snapshots of every tick, stored compactly.

A snapshot is what the website draws for one tick. Every game shapes it the same
way: a few named groups, each a mapping from a key to a JSON value, for example::

    {"bases": {"north": [1, 20]}, "fleets": {"7": [1, 12, "north", "center", 30, 80]}}

Most of a snapshot is the same from one tick to the next, so a replay keeps a
full snapshot every so often (a keyframe) and, for every tick in between, only
what changed (a delta). A delta has the same groups; a changed or new entry has
its new value and a removed entry is ``null``. The website's
``web/src/lib/frames.ts`` reads the same format.
"""

from __future__ import annotations

from typing import Any

Snapshot = dict[str, dict[str, Any]]
Delta = dict[str, dict[str, Any]]


def delta(before: Snapshot, after: Snapshot) -> Delta:
    """What changed from ``before`` to ``after``."""

    changes: Delta = {}
    for group in before.keys() | after.keys():
        old = before.get(group, {})
        new = after.get(group, {})
        changed = {key: value for key, value in new.items() if old.get(key, _MISSING) != value}
        changed.update({key: None for key in old.keys() - new.keys()})
        if changed:
            changes[group] = changed
    return changes


def apply(snapshot: Snapshot, changes: Delta) -> Snapshot:
    """A new snapshot: ``snapshot`` with ``changes`` applied."""

    result = {group: dict(entries) for group, entries in snapshot.items()}
    for group, entries in changes.items():
        target = result.setdefault(group, {})
        for key, value in entries.items():
            if value is None:
                target.pop(key, None)
            else:
                target[key] = value
    return result


class FrameRecorder:
    """Collects one snapshot per tick and stores them as keyframes and deltas."""

    def __init__(self, keyframe_every: int = 300):
        if keyframe_every < 1:
            raise ValueError("keyframe_every must be at least 1")
        self.keyframe_every = keyframe_every
        self.keyframes: dict[int, Snapshot] = {}
        self.deltas: list[Delta] = []  # deltas[t - 1] turns tick t - 1 into tick t
        self._last: Snapshot | None = None
        self._tick = -1

    def add(self, snapshot: Snapshot) -> None:
        self._tick += 1
        if self._last is not None:
            self.deltas.append(delta(self._last, snapshot))
        if self._tick % self.keyframe_every == 0:
            self.keyframes[self._tick] = snapshot
        self._last = snapshot

    def to_json(self) -> dict:
        return {
            "keyframe_every": self.keyframe_every,
            "count": self._tick + 1,
            "keyframes": {str(tick): snapshot for tick, snapshot in self.keyframes.items()},
            "deltas": self.deltas,
        }


_MISSING = object()
