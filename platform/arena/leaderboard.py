"""Standings from a tournament's results: ratings, records and head-to-head scores.

Matches with different numbers of players are different contests, so each
match size gets its own table (for Graphite: two-player and three-player maps).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations

from arena.contract import Game, Outcome
from arena.ratings import fit


def points(outcome: Outcome, seat: str) -> float:
    """1 for first place, 0 for last, evenly in between; shared places share the points."""

    size = sum(len(group) for group in outcome.ranking)
    return (size - outcome.place(seat)) / (size - 1) if size > 1 else 1.0


def standings(game: Game, results: Iterable[dict]) -> dict:
    """One table per match size, from result rows (``MatchRecord.summary()``)."""

    names = {bot.id: bot for bot in game.bots()}
    by_size: dict[int, list[dict]] = defaultdict(list)
    for row in results:
        by_size[len(row["lineup"])].append(row)

    tables = {}
    for size, rows in sorted(by_size.items()):
        outcomes = [Outcome.from_json(row["outcome"]) for row in rows]
        rankings = [
            [[row["lineup"][seat] for seat in group] for group in outcome.ranking] for row, outcome in zip(rows, outcomes)
        ]
        bots_here = {bot for row in rows for bot in row["lineup"].values()}
        anchor = game.info.rating_anchor
        ratings = fit(rankings, anchor if anchor and anchor[0] in bots_here else None)

        records = {bot: _Record() for bot in bots_here}
        head_to_head: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        for row, outcome in zip(rows, outcomes):
            for seat, bot in row["lineup"].items():
                records[bot].add(row, outcome, seat)
            for first, second in combinations(row["lineup"], 2):
                a, b = row["lineup"][first], row["lineup"][second]
                difference = outcome.place(second) - outcome.place(first)
                score = 1.0 if difference > 0 else 0.0 if difference < 0 else 0.5
                head_to_head[a][b][0] += score
                head_to_head[a][b][1] += 1
                head_to_head[b][a][0] += 1 - score
                head_to_head[b][a][1] += 1

        entries = []
        for bot in sorted(bots_here, key=lambda name: (-ratings[name].rating, name)):
            info = names.get(bot)
            entries.append(
                {
                    "bot": bot,
                    "name": info.name if info else bot,
                    "kind": info.kind if info else "",
                    "rating": round(ratings[bot].rating),
                    "interval": round(ratings[bot].interval),
                    **records[bot].to_json(),
                }
            )
        tables[str(size)] = {
            "players": size,
            "label": f"{size}-player maps",
            "matches": len(rows),
            "maps": sorted({row["map"] for row in rows}),
            "entries": entries,
            "head_to_head": {
                a: {b: {"points": score, "games": games} for b, (score, games) in sorted(row.items())}
                for a, row in sorted(head_to_head.items())
            },
        }
    return tables


class _Record:
    def __init__(self):
        self.games = self.wins = self.draws = self.losses = 0
        self.points = 0.0
        self.errors = self.refused = self.moves = 0
        self.think_ms = 0.0
        self.by_map: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])

    def add(self, row: dict, outcome: Outcome, seat: str) -> None:
        self.games += 1
        top = outcome.ranking[0] if outcome.ranking else ()
        if seat in top and len(top) == 1:
            self.wins += 1
        elif seat in top:
            self.draws += 1
        else:
            self.losses += 1
        score = points(outcome, seat)
        self.points += score
        self.by_map[row["map"]][0] += score
        self.by_map[row["map"]][1] += 1
        report = row["seats"][seat]
        self.errors += report["errors"]
        self.refused += report["refused"]
        self.moves += report["moves"]
        self.think_ms += report["think_ms_mean"] * report["moves"]

    def to_json(self) -> dict:
        return {
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "score": round(self.points / self.games, 4) if self.games else 0.0,
            "errors": self.errors,
            "refused": self.refused,
            "think_ms_mean": round(self.think_ms / self.moves, 3) if self.moves else 0.0,
            "by_map": {
                name: {"score": round(score / games, 4), "games": games} for name, (score, games) in sorted(self.by_map.items())
            },
        }
