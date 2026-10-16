"""Who plays whom, and playing it all in parallel.

Every group of bots that fits a map plays it from every seating, so no bot
gains from a better seat: on a two-player map each pair plays twice, swapping
colours; on a three-player map each trio plays all six seatings.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from itertools import combinations, permutations

from arena.contract import Game
from arena.registry import load_game, repo_root
from arena.runner import play_match


@dataclass(frozen=True)
class MatchPlan:
    match_id: str
    map: str
    lineup: tuple[tuple[str, str], ...]  # (seat, bot) in seat order
    seed: int
    showcase: bool  # keep its replay: one seating of every group on every map

    @property
    def bots(self) -> tuple[str, ...]:
        return tuple(bot for _, bot in self.lineup)


def schedule(
    game: Game,
    bots: Sequence[str] | None = None,
    maps: Sequence[str] | None = None,
    rounds: int = 1,
) -> list[MatchPlan]:
    """Every match of a full tournament, in a fixed order."""

    known_bots = {bot.id: bot for bot in game.bots()}
    known_maps = {map_info.id: map_info for map_info in game.maps()}
    chosen_bots = list(bots) if bots else [bot.id for bot in known_bots.values() if bot.leaderboard]
    chosen_maps = list(maps) if maps else list(known_maps)
    for name in chosen_bots:
        if name not in known_bots:
            raise ValueError(f"{game.info.id} has no bot {name!r}. Bots: {', '.join(known_bots)}")
    for name in chosen_maps:
        if name not in known_maps:
            raise ValueError(f"{game.info.id} has no map {name!r}. Maps: {', '.join(known_maps)}")

    plans = []
    for map_number, map_id in enumerate(chosen_maps):
        seats = known_maps[map_id].seats
        size = len(seats)
        eligible = sorted(
            name for name in chosen_bots if known_bots[name].seat_counts is None or size in known_bots[name].seat_counts
        )
        for group_number, group in enumerate(combinations(eligible, size)):
            seatings = list(permutations(group))
            shown = (map_number + group_number) % len(seatings)
            for seating_number, seating in enumerate(seatings):
                for round_number in range(rounds):
                    match_id = ".".join((map_id, *seating, str(round_number)))
                    plans.append(
                        MatchPlan(
                            match_id=match_id,
                            map=map_id,
                            lineup=tuple(zip(seats, seating)),
                            seed=_seed(match_id),
                            showcase=seating_number == shown and round_number == 0,
                        )
                    )
    return plans


def run(
    game_id: str,
    plans: Sequence[MatchPlan],
    workers: int = 1,
    keep_replay: Callable[[MatchPlan], bool] = lambda plan: plan.showcase,
) -> Iterator[tuple[MatchPlan, dict, dict | None]]:
    """Play every plan; yields (plan, result row, replay or None) as matches finish."""

    if workers <= 1:
        game = load_game(game_id)
        for plan in plans:
            yield plan, *_play(game, plan, keep_replay(plan))
        return

    # Each worker loads the game once and plays one match at a time. Neural bots
    # use one thread each, which is faster than several per match when all cores are busy.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("ARENA_ROOT", str(repo_root()))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_start_worker, initargs=(game_id,)) as pool:
        waiting = {}
        queue = list(plans)
        queue.reverse()
        while queue or waiting:
            while queue and len(waiting) < 2 * workers:
                plan = queue.pop()
                waiting[pool.submit(_play_in_worker, plan, keep_replay(plan))] = plan
            done, _ = wait(waiting, return_when=FIRST_COMPLETED)
            for future in done:
                plan = waiting.pop(future)
                yield plan, *future.result()


def _seed(match_id: str) -> int:
    return int.from_bytes(hashlib.sha256(match_id.encode()).digest()[:4], "big")


def _play(game: Game, plan: MatchPlan, record: bool) -> tuple[dict, dict | None]:
    record_ = play_match(game, plan.map, dict(plan.lineup), plan.seed, plan.match_id, record_replay=record)
    return record_.summary(), record_.replay


_worker_game: Game | None = None


def _start_worker(game_id: str) -> None:
    global _worker_game
    _worker_game = load_game(game_id)


def _play_in_worker(plan: MatchPlan, record: bool) -> tuple[dict, dict | None]:
    return _play(_worker_game, plan, record)
