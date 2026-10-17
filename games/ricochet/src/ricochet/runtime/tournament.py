from __future__ import annotations

import csv
import itertools
import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from ricochet.game.config import RulesConfig
from ricochet.game.mapgen import generate_map
from ricochet.game.types import MapDefinition
from ricochet.runtime.loading import available_ais
from ricochet.runtime.match import MatchSummary, run_headless

# One game: (first seat, second seat, map, match seed, rules).
Fixture = tuple[str, str, str | MapDefinition, int, RulesConfig | None]
Progress = Callable[[int, int], None]


def _play(fixture: Fixture) -> MatchSummary:
    first, second, map_id, seed, rules = fixture
    return run_headless(first, second, map_id, seed, rules)


def _play_batch(fixtures: Sequence[Fixture]) -> list[MatchSummary]:
    return [_play(fixture) for fixture in fixtures]


def play_fixtures(
    fixtures: Sequence[Fixture],
    workers: int | None = 1,
    on_result: Progress | None = None,
) -> list[MatchSummary]:
    """Play every fixture, in parallel processes when ``workers`` > 1 (``None``: all CPUs).

    Each game is deterministic, and results keep fixture order, so the output does not
    depend on the number of workers.
    """
    workers = workers or os.cpu_count() or 1
    total = len(fixtures)
    if workers == 1 or total < 2:
        results = []
        for fixture in fixtures:
            results.append(_play(fixture))
            if on_result is not None:
                on_result(len(results), total)
        return results
    # Small batches keep progress smooth; they are reassembled in fixture order.
    size = max(1, min(16, total // (workers * 4)))
    batches = [fixtures[start : start + size] for start in range(0, total, size)]
    finished: list[list[MatchSummary]] = [[] for _ in batches]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_play_batch, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            index = futures[future]
            finished[index] = future.result()
            done += len(finished[index])
            if on_result is not None:
                on_result(done, total)
    return [summary for batch in finished for summary in batch]


def _check_bots(bots: Sequence[str]) -> None:
    if len(bots) < 2:
        raise ValueError("a tournament needs at least two bots")
    known = available_ais()
    unknown = [bot for bot in bots if not bot.startswith("checkpoint:") and bot not in known]
    if unknown:
        raise ValueError(f"unknown bots {', '.join(unknown)}; available: {', '.join(known)}")


def run_tournament(
    bots: tuple[str, ...],
    maps: tuple[str, ...],
    seeds: tuple[int, ...],
    workers: int | None = 1,
    on_result: Progress | None = None,
) -> list[MatchSummary]:
    """Every pairing on every listed map and seed, from both seats."""
    _check_bots(bots)
    fixtures: list[Fixture] = []
    for first, second in itertools.combinations(bots, 2):
        for map_id in maps:
            for seed in seeds:
                fixtures.append((first, second, map_id, seed, None))
                fixtures.append((second, first, map_id, seed, None))
    return play_fixtures(fixtures, workers, on_result)


def run_generated_tournament(
    bots: tuple[str, ...],
    rounds: int = 100,
    seed: int = 0,
    workers: int | None = None,
    rules: RulesConfig | None = None,
    on_result: Progress | None = None,
) -> list[MatchSummary]:
    """Round robin on freshly generated maps: each pairing plays ``rounds`` rounds.

    Round ``r`` uses map and match seed ``seed + r`` and is played from both seats, so every
    pairing sees the same maps and the slight map asymmetry cannot favor either bot.
    """
    _check_bots(bots)
    if rounds < 1:
        raise ValueError("rounds must be positive")
    maps = [generate_map(seed + offset, rules=rules) for offset in range(rounds)]
    fixtures: list[Fixture] = []
    for first, second in itertools.combinations(bots, 2):
        for offset, definition in enumerate(maps):
            fixtures.append((first, second, definition, seed + offset, rules))
            fixtures.append((second, first, definition, seed + offset, rules))
    return play_fixtures(fixtures, workers, on_result)


def write_results(results: list[MatchSummary], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([result.to_dict() for result in results], indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    rows = []
    for result in results:
        row = asdict(result)
        row["p1"] = result.controllers[0]
        row["p2"] = result.controllers[1]
        row["p1_health"], row["p2_health"] = result.health
        row["p1_faults"], row["p2_faults"] = result.faults
        row["p1_mean_action_ms"], row["p2_mean_action_ms"] = result.mean_action_ms
        row["p1_max_action_ms"], row["p2_max_action_ms"] = result.max_action_ms
        for key in ("controllers", "health", "faults", "mean_action_ms", "max_action_ms"):
            row.pop(key)
        rows.append(row)
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def standings(results: list[MatchSummary]) -> list[tuple[str, float, int, int, int]]:
    table: dict[str, list[float | int]] = {}
    for result in results:
        if result.invalid:
            continue
        for bot in result.controllers:
            table.setdefault(bot, [0.0, 0, 0, 0])
        if result.winner is None:
            for bot in result.controllers:
                table[bot][0] += 0.5
                table[bot][2] += 1
        else:
            winner = result.controllers[result.winner]
            loser = result.controllers[1 - result.winner]
            table[winner][0] += 1.0
            table[winner][1] += 1
            table[loser][3] += 1
    rows = (
        (bot, float(values[0]), int(values[1]), int(values[2]), int(values[3]))
        for bot, values in table.items()
    )
    return sorted(rows, key=lambda row: (-row[1], row[0]))


def head_to_head(results: list[MatchSummary]) -> dict[tuple[str, str], tuple[int, int, int]]:
    """``(bot, opponent) -> (wins, draws, losses)`` of ``bot``, over valid games."""
    table: dict[tuple[str, str], list[int]] = {}
    for result in results:
        if result.invalid:
            continue
        for seat in (0, 1):
            pair = (result.controllers[seat], result.controllers[1 - seat])
            record = table.setdefault(pair, [0, 0, 0])
            if result.winner is None:
                record[1] += 1
            else:
                record[0 if result.winner == seat else 2] += 1
    return {pair: (wins, draws, losses) for pair, (wins, draws, losses) in table.items()}
