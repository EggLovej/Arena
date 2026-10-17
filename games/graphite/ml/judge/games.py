"""Step 2 of the judge loop: play many games and keep positions labelled with how they ended.

The player (by default the strongest network) plays itself, and plays a set
of other opponents so the judge also learns how games against other styles
go. Every seat is recorded, also the opponents', from that seat's point of
view and labelled with that seat's final result. Neighbouring ticks are
nearly the same position, so only ``positions_per_seat`` positions per seat
and game are kept.

Games are split, whole, into training, held-out (to watch training), and
test (for the final verdict), so no game has positions on two sides.

Maps come from the training seeds (below the evaluation seeds), so the
judge is never tested on maps it saw.

    .venv/bin/python -m ml judge-games --name round1
    .venv/bin/python -m ml judge-games --name round1 --resume      carry on after stopping
    .venv/bin/python -m ml judge-games --name small --self-games 200 --games-per-opponent 20

Everything lands in ``ml/runs/judge_data/NAME/``.
"""

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from ml.evaluation.evaluate import FIRST_EVALUATION_SEED
from ml.evaluation.saved_games import short_name
from ml.judge.progress import ProgressBlock, count, duration
from ml.map_generator import generate_map
from ml.models.checkpoint import load_checkpoint
from ml.play.rollout import Matchup, RolloutPool, Seat, Trajectory
from ml.training.runlog import RUNS_DIRECTORY


DATA_DIRECTORY = RUNS_DIRECTORY / "judge_data"
SPLITS = ("training", "held_out", "test")
SELF = "itself"


@dataclass(frozen=True)
class GameSettings:
    player: str = "ml/runs/deep_from_imitation/round_0500.pt"
    self_games: int = 10_000
    games_per_opponent: int = 500
    opponents: tuple[str, ...] = (
        "aggressive_ai",
        "careful_ai",
        "strategic_ai",
        "ml/runs/deep_scratch/round_0300.pt",
        "ml/runs/overnight_scratch/round_0500.pt",
    )
    positions_per_seat: int = 20
    held_out_share: float = 0.1
    test_share: float = 0.1
    max_seconds: float = 180.0
    games_per_chunk: int = 96  # games played between two saves (progress is shown game by game regardless)
    seed: int = 0
    workers: int = 6


def generate(name: str, settings: GameSettings = GameSettings(), resume: bool = False) -> Path:
    """Play the games and save the positions. Returns the data folder.

    The groups are played one after the other: all games against itself,
    then all against the first opponent, and so on. The progress shows every
    finished game as it happens.
    """

    folder = DATA_DIRECTORY / name
    state_file = folder / "progress.json"
    if resume:
        state = json.loads(state_file.read_text())
        settings = GameSettings(**{**asdict(settings), **_settings_from(state["settings"])})
    else:
        if state_file.exists():
            raise FileExistsError(f"Data '{name}' already exists in {folder}; pick another name or pass --resume")
        # The seed picks the maps. Two data runs with one seed play the same maps, so one run's
        # test games would share their maps with the other's training games.
        for other in DATA_DIRECTORY.glob("*/progress.json"):
            if json.loads(other.read_text())["settings"]["seed"] == settings.seed:
                raise ValueError(
                    f"Data '{other.parent.name}' already used seed {settings.seed}, so it would play the same maps. "
                    f"Pick another, e.g. --seed {settings.seed + 1}."
                )
        (folder / "parts").mkdir(parents=True, exist_ok=True)
        state = {
            "settings": asdict(settings),
            "games": {group: 0 for group in _groups(settings)},
            "seconds": {group: 0.0 for group in _groups(settings)},  # time spent on each group
            "positions": {split: 0 for split in SPLITS},
            "chunks": 0,
        }
        state_file.write_text(json.dumps(state, indent=1))

    targets = _groups(settings)
    networks = {settings.player, *(o for o in settings.opponents if o.endswith(".pt"))}
    models = {path: load_checkpoint(path).model for path in networks}
    block = ProgressBlock("Generating games")
    block.show(_lines(state, targets, None))
    pool = RolloutPool(settings.workers)
    try:
        for group, target in targets.items():
            group_started = time.monotonic() - state["seconds"][group]
            while state["games"][group] < target:
                rng = np.random.default_rng([settings.seed, state["chunks"]])
                size = min(settings.games_per_chunk, target - state["games"][group])
                matchups = [_matchup(group, settings, rng, state["games"][group] + number) for number in range(size)]
                splits = [_split(rng, settings) for _ in matchups]
                # Positions as they will be kept, counted live; the chunk is saved when it is complete.
                live = {"games": 0, "positions": dict.fromkeys(SPLITS, 0)}

                def finished(index: int, decisions: list[int], winner: int | None) -> None:
                    live["games"] += 1
                    live["positions"][splits[index]] += sum(min(n, settings.positions_per_seat) for n in decisions)
                    state["seconds"][group] = time.monotonic() - group_started
                    block.show(_lines(state, targets, group, live))

                records = pool.play(
                    matchups, models, seed=int(rng.integers(2**31)), max_seconds=settings.max_seconds, on_finished=finished
                )
                kept = {split: [] for split in SPLITS}
                for split, record in zip(splits, records):
                    kept[split] += [_sample(trajectory, settings.positions_per_seat, rng) for trajectory in record.trajectories]
                for split, trajectories in kept.items():
                    if trajectories:
                        path = folder / "parts" / f"{split}_{state['chunks']:05d}.pkl"
                        path.write_bytes(pickle.dumps(trajectories, protocol=pickle.HIGHEST_PROTOCOL))
                        state["positions"][split] += sum(len(t) for t in trajectories)
                state["games"][group] += size
                state["chunks"] += 1
                state["seconds"][group] = time.monotonic() - group_started
                state_file.write_text(json.dumps(state, indent=1))
                block.show(_lines(state, targets, group))
    except KeyboardInterrupt:
        # Workers in the middle of a game would otherwise keep the process waiting.
        pool._pool.terminate()
        raise
    pool.close()

    block.finish(_lines(state, targets, None))
    print(summary(folder), flush=True)
    return folder


def summary(folder: Path) -> str:
    """Total time, games, positions, and size on disk of a finished (or stopped) data run."""

    state = json.loads((folder / "progress.json").read_text())
    size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    positions = sum(state["positions"].values())
    return (
        f"Total: {count(sum(state['games'].values()))} games, {count(positions)} positions, "
        f"{duration(sum(state['seconds'].values()))} of playing, {size / 1e6:,.0f} MB on disk "
        f"({size / max(positions, 1) / 1e3:.1f} KB per position)\nSaved to {folder}"
    )


def load_positions(name: str | Path, split: str) -> list[Trajectory]:
    """Every kept position of one split, as short trajectories (one per seat and game)."""

    folder = Path(name) if Path(name).is_dir() else DATA_DIRECTORY / str(name)
    trajectories = []
    for path in sorted((folder / "parts").glob(f"{split}_*.pkl")):
        trajectories += pickle.loads(path.read_bytes())
    return trajectories


def _groups(settings: GameSettings) -> dict[str, int]:
    return {SELF: settings.self_games, **{opponent: settings.games_per_opponent for opponent in settings.opponents}}


def _matchup(group: str, settings: GameSettings, rng: np.random.Generator, number: int) -> Matchup:
    definition = generate_map(int(rng.integers(FIRST_EVALUATION_SEED))).definition
    player = Seat(settings.player, record=True)
    if group == SELF:
        return Matchup(definition, player, player)
    opponent = Seat(group, record=True)
    # Every other game the player takes the other seat.
    return Matchup(definition, player, opponent) if number % 2 == 0 else Matchup(definition, opponent, player)


def _split(rng: np.random.Generator, settings: GameSettings) -> str:
    draw = rng.random()
    if draw < settings.test_share:
        return "test"
    if draw < settings.test_share + settings.held_out_share:
        return "held_out"
    return "training"


def _sample(trajectory: Trajectory, positions: int, rng: np.random.Generator) -> Trajectory:
    """The same trajectory, with only a random sample of its positions (in order)."""

    if len(trajectory) <= positions:
        keep = np.arange(len(trajectory))
    else:
        keep = np.sort(rng.choice(len(trajectory), size=positions, replace=False))
    per_position = ("base_features", "global_features", "route_features", "allowed_bits", "source", "target", "amount", "tick")
    changes = {name: getattr(trajectory, name)[keep] for name in per_position}
    for name in ("log_probability", "value", "captured"):
        if getattr(trajectory, name) is not None:
            changes[name] = getattr(trajectory, name)[keep]
    return replace(trajectory, **changes)


def _lines(state: dict, targets: dict[str, int], current: str | None, live: dict | None = None) -> list[str]:
    """One line per group: finished groups with the time they took, the current one with the time it still needs."""

    width = len(count(max(targets.values())))
    lines = []
    for group, target in targets.items():
        done = state["games"][group] + (live["games"] if live and group == current else 0)
        seconds = state["seconds"][group]
        label = "vs itself" if group == SELF else f"vs {short_name(group)}"
        line = f"({count(done):>{width}} / {count(target):>{width}}) {label:28}"
        if done >= target:
            line += f"took {duration(seconds)}"
        elif group == current and done:
            line += f"{duration(seconds / done * (target - done))} remaining   ({seconds / done:.2f}s per game)"
        elif group == current:
            line += "starting"
        lines.append(line)
    positions = dict(state["positions"])
    if live:
        for split, number in live["positions"].items():
            positions[split] += number
    lines.append(
        f"positions recorded: {count(sum(positions.values()))} "
        f"(training {count(positions['training'])} / held out {count(positions['held_out'])} / test {count(positions['test'])})"
    )
    return lines


def _settings_from(saved: dict) -> dict:
    known = {setting.name for setting in fields(GameSettings)}
    return {key: tuple(value) if isinstance(value, list) else value for key, value in saved.items() if key in known}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="Folder name in ml/runs/judge_data/")
    parser.add_argument("--resume", action="store_true", help="Carry on an earlier run of the same name, with its settings")
    defaults = GameSettings()
    for setting in fields(GameSettings):
        option = "--" + setting.name.replace("_", "-")
        default = getattr(defaults, setting.name)
        if isinstance(default, tuple):
            parser.add_argument(option, nargs="+", default=list(default), metavar="NAME", help=f"default: {' '.join(default)}")
        else:
            parser.add_argument(option, type=type(default), default=default, metavar="N", help=f"default: {default}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = {setting.name: getattr(args, setting.name) for setting in fields(GameSettings)}
    values["opponents"] = tuple(values["opponents"])
    try:
        generate(args.name, GameSettings(**values), resume=args.resume)
    except KeyboardInterrupt:
        print(f"\nStopped. Carry on with: python -m ml judge-games --name {args.name} --resume", file=sys.stderr)


if __name__ == "__main__":
    main()
