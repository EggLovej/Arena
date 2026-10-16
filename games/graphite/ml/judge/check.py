"""Step 4 of the judge loop: does the look-ahead play better with the new judge?

The look-ahead keeps proposing moves with the network's policy output; only
the judging of the table changes. Three tests, on fresh evaluation maps:

1. **Head to head:** the look-ahead with the new judge against the same
   look-ahead with the old one (the network's own value output), and
   against Beast (our strongest player so far), if Beast is someone else.
2. **Fixed references:** the new one against the original Atlas (its
   opponent network pinned to the same network) and the plain network.
3. **Rematches:** every saved loss in ``ml/runs/losses/`` started again from
   the seat that lost, once with the new judge and once with the old one.
   Starts the new judge still loses are saved as losses of their own, to
   watch with ``python -m ml watch``.

    .venv/bin/python -m ml judge-check --judge ml/runs/judge1/best.pt
    .venv/bin/python -m ml judge-check --judge ml/runs/judge1/best.pt --games 200 --reference-games 100

The results are also written next to the judge, as ``check.json``.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

from ml.evaluation.evaluate import Evaluation, _prepare_worker, evaluate
from ml.evaluation.rematch import play_again
from ml.evaluation.saved_games import load_game, saved_games
from ml.judge.progress import ProgressBlock, duration

NETWORK = "ml/runs/deep_from_imitation/round_0500.pt"


class Board:
    """Live counts for several sections of matches, shown as one block."""

    def __init__(self, title: str, sections: dict[str, int], workers: int | None):
        self.sections = {name: {"total": total, "results": []} for name, total in sections.items()}
        self.workers = workers
        self.block = ProgressBlock(title)
        self.started = time.monotonic()
        self.show()

    def show(self, final: bool = False) -> None:
        lines = []
        for title, section in self.sections.items():
            if section["total"]:
                results = section["results"]
                wins = sum(outcome == "win" for outcome in results)
                draws = sum(outcome == "draw" for outcome in results)
                score = f"   {(wins + draws / 2) / len(results):6.1%}" if results else ""
                lines.append(
                    f"{title:32} ({len(results):3d} / {section['total']:3d}):  "
                    f"{wins} wins / {draws} draws / {len(results) - wins - draws} losses{score}"
                )
        lines.append(f"{duration(time.monotonic() - self.started)} so far")
        (self.block.finish if final else self.block.show)(lines)

    def matches(self, title: str, candidate: str, opponent: str) -> None:
        """Play the section's games on fresh evaluation maps, both seats, counting as they finish."""

        def progress(results, _):
            self.sections[title]["results"] = [result.outcome for result in results]
            self.show()

        evaluation: Evaluation = evaluate(
            candidate, [opponent], generated_maps=max(1, self.sections[title]["total"] // 2), bundled_maps=False,
            skip_generated=30, workers=self.workers, on_progress=progress,
        )
        self.sections[title]["results"] = [result.outcome for result in evaluation.results]

    def add(self, title: str, outcome: str) -> None:
        self.sections[title]["results"].append(outcome)
        self.show()

    def summary(self) -> dict:
        return {
            title: {
                "games": len(section["results"]),
                "wins": sum(o == "win" for o in section["results"]),
                "draws": sum(o == "draw" for o in section["results"]),
                "losses": sum(o == "loss" for o in section["results"]),
            }
            for title, section in self.sections.items()
        }


def check_judge(
    judge: str,
    network: str = NETWORK,
    rule: str = "nash",
    games: int = 100,
    reference_games: int = 60,
    rematches: bool = True,
    workers: int | None = None,
    atlas_network: str = NETWORK,
) -> dict:
    new = f"lookahead:{network}?rule={rule}&judge={judge}"
    old = f"lookahead:{network}?rule={rule}"
    # The original Atlas forecasts a network opponent with this file. Its frozen copy of our encoders
    # only reads networks saved before the encoders grew (deep_from_imitation round 500 and older).
    os.environ["ATLAS_OPPONENT_CHECKPOINT"] = atlas_network

    from ml.play.beast import current

    beast = current()
    against_beast = beast is not None and beast["player"] != new
    lost_starts = distinct_starts() if rematches else []
    board = Board(
        "Evaluation",
        {
            "head to head vs old judge": games,
            **({"vs Beast": games} if against_beast else {}),
            "vs Atlas": reference_games,
            "vs plain network": reference_games,
            "rematches, new judge": len(lost_starts),
            "rematches, old judge": len(lost_starts),
        },
        workers,
    )
    board.matches("head to head vs old judge", new, old)
    if against_beast:
        board.matches("vs Beast", new, beast["player"])
    board.matches("vs Atlas", new, "atlas_ai")
    board.matches("vs plain network", new, network)
    saved_losses: list[str] = []
    for title, player in (("rematches, new judge", new), ("rematches, old judge", old)):
        jobs = [(game, player, game.candidate_seat, game.seed, player == new) for game in lost_starts]
        with Pool(workers or max(1, (os.cpu_count() or 2) - 2), initializer=_prepare_worker) as pool:
            for outcome, _, _, path in pool.imap_unordered(_rematch, jobs):
                board.add(title, outcome)
                if path:
                    saved_losses.append(path)
    board.show(final=True)
    if saved_losses:
        print(f"{len(saved_losses)} starts the new judge still lost are saved; list them with: python -m ml watch", flush=True)

    summary = board.summary()
    summary["judge"], summary["network"], summary["rule"] = judge, network, rule
    summary["beast"] = beast["player"] if beast else None
    target = Path(judge).with_name("check.json")
    target.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"Results saved to {target}", flush=True)
    return summary


def check_policy(
    policy: str,
    judge: str,
    old_network: str = NETWORK,
    rule: str = "nash",
    games: int = 100,
    reference_games: int = 60,
    workers: int | None = None,
) -> dict:
    """Did teaching help? The new network on instinct against the old one, and with the look-ahead against Beast and Atlas."""

    from ml.play.beast import current

    beast = current()
    new_lookahead = f"lookahead:{policy}?rule={rule}&judge={judge}"
    # Atlas's frozen copy of the encoders only reads checkpoints saved before the input features
    # grew; a newly saved network is refused. Its forecast uses the network the policy was taught from.
    os.environ["ATLAS_OPPONENT_CHECKPOINT"] = old_network
    sections = {"on instinct: new vs old network": games}
    if beast:
        sections["look-ahead: new vs Beast"] = games
    sections["look-ahead: new vs Atlas"] = reference_games
    board = Board("Evaluation", sections, workers)
    board.matches("on instinct: new vs old network", policy, old_network)
    if beast:
        board.matches("look-ahead: new vs Beast", new_lookahead, beast["player"])
    board.matches("look-ahead: new vs Atlas", new_lookahead, "atlas_ai")
    board.show(final=True)

    summary = board.summary()
    summary.update({"policy": policy, "judge": judge, "old network": old_network, "beast": beast["player"] if beast else None})
    target = Path(policy).with_name("check.json")
    target.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"Results saved to {target}", flush=True)
    return summary


def distinct_starts():
    """Each saved lost start once: the same map, seat, and opponent saved by several players is one start."""

    starts = {}
    for path in saved_games():
        game = load_game(path)
        starts.setdefault((game.map_source, game.candidate_seat, game.opponent), game)
    return list(starts.values())


def _rematch(job):
    return play_again(*job)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge", required=True, help="The judge checkpoint, e.g. ml/runs/judge1/best.pt")
    parser.add_argument("--network", default=NETWORK, help="The network whose policy proposes the moves")
    parser.add_argument("--rule", default="nash", choices=("nash", "expected", "worst"))
    parser.add_argument("--games", type=int, default=100, help="Head-to-head games (default 100)")
    parser.add_argument("--reference-games", type=int, default=60, help="Games against each fixed reference (default 60)")
    parser.add_argument("--no-rematches", action="store_true", help="Skip replaying the saved losses")
    parser.add_argument(
        "--atlas-network", default=NETWORK,
        help="The network Atlas forecasts us with; it must predate the grown encoders (default: %(default)s)",
    )
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_judge(
        args.judge, args.network, args.rule, args.games, args.reference_games, not args.no_rematches, args.workers,
        args.atlas_network,
    )


def parse_policy_args():
    parser = argparse.ArgumentParser(description="Did teaching help? The taught network against the old one, Beast, and Atlas.")
    parser.add_argument("--policy", required=True, help="The taught network, e.g. ml/runs/policy1/best.pt")
    parser.add_argument("--judge", required=True, help="The judge its look-ahead uses")
    parser.add_argument("--old-network", default=NETWORK, help="The network it was taught from")
    parser.add_argument("--rule", default="nash", choices=("nash", "expected", "worst"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--reference-games", type=int, default=60)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def policy_main() -> None:
    args = parse_policy_args()
    check_policy(args.policy, args.judge, args.old_network, args.rule, args.games, args.reference_games, args.workers)


if __name__ == "__main__":
    main()
