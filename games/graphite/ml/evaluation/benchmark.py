"""How fast this machine trains: the parts of a self-play round, timed.

A round is two halves: playing (CPU workers simulate the games; the network
decides, on each worker's CPU or in one GPU server) and learning (the GPU).
This command times both, for a few network sizes, so two machines (this Mac,
a Colab runtime) can be compared with the same code:

1. the hardware,
2. one forward pass of each network: one CPU thread, and the GPU at two batch sizes,
3. one round of 96 self-play games with each way of running the network,
4. one round of learning (4 passes over its decisions, as in training).

Playing time depends on how long the games last, and untrained networks play
longer games than trained ones; so it is also given per 1,000 decisions and
for a typical round of the trained 64x6 network (175,000 decisions).

    .venv/bin/python -m ml benchmark
    .venv/bin/python -m ml benchmark --sizes 64x6 128x8 256x8 --workers 6 11

The last line of the output is the result as JSON (``BENCHMARK_JSON {...}``).
"""

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ml.device import default_device
from ml.map_generator import generate_map
from ml.models import ModelInput, build_model, make_input
from ml.models.checkpoint import load_checkpoint
from ml.play.rollout import Matchup, RolloutPool, Seat
from ml.training.selfplay import SelfPlaySettings, judged_decisions, update

# The trained network to time as 64x6, when it is there; otherwise random weights.
TRAINED = "ml/runs/shaping_entropy/round_0150.pt"
TYPICAL_ROUND = 175_000  # decisions in a round of the trained 64x6 network


def hardware() -> dict:
    info = {
        "cpu cores": os.cpu_count(),
        "cpu": _cpu_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": None,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        info["gpu"] = "Apple GPU (mps)"
    try:
        info["memory GB"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        pass
    return info


def _cpu_name() -> str:
    for command in (["sysctl", "-n", "machdep.cpu.brand_string"], ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo"]):
        try:
            text = subprocess.run(command, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            continue
        if text:
            return text.split(":", 1)[-1].strip()
    return platform.processor() or "unknown"


def network(size: str):
    if size == "64x6" and Path(TRAINED).exists():
        return load_checkpoint(TRAINED).model, "trained"
    width, layers = (int(part) for part in size.split("x"))
    return build_model(width=width, layers=layers), "random weights"


def _positions(count: int = 40) -> ModelInput:
    from game import Game
    from game_types import RED

    from ml.encoding.action_encoder import legal_actions
    from ml.encoding.state_encoder import encode_state

    views = []
    for seed in range(count):
        game = Game(generate_map(seed).definition)
        for _ in range(300):
            game.step({})
        views.append(game.view_for(RED))
    return make_input([encode_state(view) for view in views], [legal_actions(view) for view in views])


def _batch(base: ModelInput, size: int, device) -> ModelInput:
    repeats = -(-size // base.base_mask.shape[0])
    values = (getattr(base, name) for name in ModelInput.__dataclass_fields__)
    return ModelInput(*(None if value is None else torch.cat([value] * repeats)[:size] for value in values)).to(device)


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def forward_times(model, base: ModelInput, device) -> dict:
    """Milliseconds per batch: one CPU thread at 32 positions, the GPU at 192 and 768."""

    times = {}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = model.cpu().eval()
    x = _batch(base, 32, "cpu")
    with torch.no_grad():
        model(x)
        started = time.perf_counter()
        for _ in range(10):
            model(x)
    times["cpu, 1 thread, 32 positions"] = (time.perf_counter() - started) / 10 * 1000
    torch.set_num_threads(threads)
    if device.type != "cpu":
        model = model.to(device)
        for size in (192, 768):
            x = _batch(base, size, device)
            with torch.no_grad():
                for _ in range(3):
                    model(x)
                _sync(device)
                started = time.perf_counter()
                for _ in range(20):
                    model(x)
                _sync(device)
            times[f"gpu, {size} positions"] = (time.perf_counter() - started) / 20 * 1000
        model.cpu()
    return times


def round_times(model, workers: int, inference_device, games: int, device) -> tuple[dict, list]:
    """One round of self-play games: seconds, decisions, and the records (for timing the learning)."""

    rng = np.random.default_rng(1)
    seat = Seat("network", record=True)
    matchups = [Matchup(generate_map(int(rng.integers(900_000))).definition, seat, seat) for _ in range(games)]
    model = model.to(device)
    with RolloutPool(workers, inference_device=inference_device) as pool:
        pool.play(matchups[: workers * 2], {"network": model}, seed=1, max_seconds=10)  # start-up and first shapes
        started = time.perf_counter()
        records = pool.play(matchups, {"network": model}, seed=1)
        seconds = time.perf_counter() - started
    decisions = sum(len(trajectory) for record in records for trajectory in record.trajectories)
    return {
        "seconds": seconds,
        "decisions": decisions,
        "seconds per 1000 decisions": seconds / decisions * 1000,
        "seconds for a typical round": seconds / decisions * TYPICAL_ROUND,
    }, records


def learning_time(model, records, device) -> float:
    """Seconds for one round of learning: 4 passes over a sample of the round's decisions."""

    settings = SelfPlaySettings(material_hint=0, max_policy_change=1e9)
    rng = np.random.default_rng(0)
    decisions, _ = judged_decisions(records, settings, 0.0, rng, model.free_amounts)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    update(model, optimizer, decisions, settings, rng, device)  # first shapes
    _sync(device)
    started = time.perf_counter()
    update(model, optimizer, decisions, settings, rng, device)
    _sync(device)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=["64x6", "128x8"], help="Network sizes, width x layers")
    parser.add_argument("--workers", nargs="+", type=int, help="Worker counts to try (default: half the cores, and all but one)")
    parser.add_argument("--games", type=int, default=96)
    args = parser.parse_args()

    device = default_device()
    cores = os.cpu_count() or 2
    worker_counts = args.workers or sorted({max(1, cores // 2), max(1, cores - 1)})
    result = {"hardware": hardware(), "sizes": {}}
    print("Hardware: " + ", ".join(f"{k} {v}" for k, v in result["hardware"].items()), flush=True)
    base = _positions()

    for size in args.sizes:
        model, kind = network(size)
        weights = sum(p.numel() for p in model.parameters())
        entry = {"weights": weights, "kind": kind, "forward ms": forward_times(model, base, device), "rounds": {}}
        print(f"\n{size} ({kind}, {weights:,} weights): forward " + ", ".join(f"{k} {v:.1f} ms" for k, v in entry["forward ms"].items()), flush=True)
        records = None
        modes = [("workers", None)] + ([("server", device)] if device.type != "cpu" else [])
        for workers in worker_counts:
            for mode, inference_device in modes:
                timing, played = round_times(model, workers, inference_device, args.games, device)
                entry["rounds"][f"{mode}, {workers} workers"] = timing
                records = records or played
                print(
                    f"  playing, {mode:7s} {workers:2d} workers: {timing['seconds']:6.1f} s for {timing['decisions']:,} decisions"
                    f"  ({timing['seconds per 1000 decisions']:.2f} s per 1000; a typical round {timing['seconds for a typical round']:.0f} s)",
                    flush=True,
                )
        entry["learning seconds"] = learning_time(model, records, device)
        best = min(entry["rounds"].items(), key=lambda item: item[1]["seconds for a typical round"])
        entry["typical round seconds"] = best[1]["seconds for a typical round"] + entry["learning seconds"]
        entry["best way to play"] = best[0]
        print(
            f"  learning ({device.type}): {entry['learning seconds']:.1f} s"
            f"  ->  a typical round about {entry['typical round seconds']:.0f} s (playing with {best[0]})",
            flush=True,
        )
        result["sizes"][size] = entry
    print("\nBENCHMARK_JSON " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
