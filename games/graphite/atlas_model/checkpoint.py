"""Save a model to a file and load it back.

A checkpoint is the only thing trainers hand to each other. It holds:

- which architecture it is, and the settings it was built with
- the weights
- its history: one entry per step that produced it, such as a random start,
  then imitation, then self-play
- the feature lists the encoders had when it was saved

A model only makes sense with the inputs it was trained on, so loading is
refused if the encoders have changed since.

From the game directory:

    .venv/bin/python -m ml.checkpoint new ml/runs/demo/random.pt --seed 0
    .venv/bin/python -m ml.checkpoint show ml/runs/demo/random.pt
"""

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from atlas_model.encoding.action_encoder import AMOUNT_OPTIONS
from atlas_model.encoding.state_encoder import BASE_FEATURES, GLOBAL_FEATURES, PAIR_FEATURES
from atlas_model.models import DEFAULT_MODEL, MODELS, PolicyValueModel, build_model, name_of


FORMAT_VERSION = 1
RANDOM = "random"


@dataclass
class Checkpoint:
    model: PolicyValueModel
    model_name: str
    # Oldest first. Each entry is a small dictionary with at least "step" and "time".
    history: list[dict] = field(default_factory=list)
    # Anything a trainer needs to carry on exactly where it stopped, such as optimizer state.
    training_state: dict | None = None


def init_model(
    init: str | Path = RANDOM,
    model_name: str = DEFAULT_MODEL,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    **settings,
) -> Checkpoint:
    """The one place a trainer gets its starting model from.

    ``init`` is "random" for fresh weights, or the path of a checkpoint to
    continue from. ``model_name``, ``seed``, and ``settings`` only apply to a
    random start; a checkpoint already knows its own architecture.
    """

    if str(init) != RANDOM:
        return load_checkpoint(init, device)

    # Seeding inside fork_rng leaves the rest of the program's randomness untouched.
    with torch.random.fork_rng():
        if seed is not None:
            torch.manual_seed(seed)
        model = build_model(model_name, **settings)
    entry = history_entry("random start", model=model_name, seed=seed)
    return Checkpoint(model.to(device), model_name, [entry])


def save_checkpoint(
    path: str | Path,
    model: PolicyValueModel,
    history: list[dict] | None = None,
    training_state: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = {
        "format_version": FORMAT_VERSION,
        "model_name": name_of(model),
        "model_config": asdict(model.config),
        "weights": {name: weight.detach().cpu() for name, weight in model.state_dict().items()},
        "history": list(history or []),
        "training_state": training_state,
        "encoding": _current_encoding(),
    }
    # Write beside the target, then swap it in, so an interrupted save never leaves half a file.
    temporary = path.with_name(path.name + ".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Checkpoint:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    # weights_only refuses files that contain code, so a checkpoint cannot run anything on load.
    contents = torch.load(path, map_location="cpu", weights_only=True)

    if contents.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} uses checkpoint format {contents.get('format_version')}, expected {FORMAT_VERSION}")
    model_name = contents["model_name"]
    if model_name not in MODELS:
        raise ValueError(f"{path} holds a '{model_name}' model, which is not listed in ml/models")

    changed = [
        name
        for name, saved in contents["encoding"].items()
        if list(saved) != _current_encoding().get(name)
    ]
    if changed:
        raise ValueError(
            f"{path} was trained with different {', '.join(changed)} than the encoders produce now. "
            "Retrain it, or restore the encoders it was trained with."
        )

    model = build_model(model_name, **contents["model_config"])
    model.load_state_dict(contents["weights"])
    return Checkpoint(
        model=model.to(device),
        model_name=model_name,
        history=contents["history"],
        training_state=contents["training_state"],
    )


def history_entry(step: str, **details) -> dict:
    """One line of a checkpoint's history, stamped with the current time."""

    return {"step": step, "time": datetime.now().isoformat(timespec="seconds"), **details}


def describe(checkpoint: Checkpoint) -> str:
    model = checkpoint.model
    lines = [
        f"Model: {checkpoint.model_name}, {sum(p.numel() for p in model.parameters()):,} weights",
        "Settings: " + ", ".join(f"{name}={value}" for name, value in asdict(model.config).items()),
        "History:",
    ]
    for number, entry in enumerate(checkpoint.history, start=1):
        details = ", ".join(f"{name}={value}" for name, value in entry.items() if name not in ("step", "time"))
        lines.append(f"  {number}. {entry['step']} ({entry['time']})" + (f": {details}" if details else ""))
    if not checkpoint.history:
        lines.append("  (none recorded)")
    return "\n".join(lines)


def _current_encoding() -> dict[str, list[str]]:
    return {
        "base_features": list(BASE_FEATURES),
        "pair_features": list(PAIR_FEATURES),
        "global_features": list(GLOBAL_FEATURES),
        "amount_options": list(AMOUNT_OPTIONS),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Create or inspect a checkpoint.")
    commands = parser.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="Save a model with fresh random weights")
    new.add_argument("path")
    new.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    new.add_argument("--seed", type=int)

    show = commands.add_parser("show", help="Print what a checkpoint contains")
    show.add_argument("path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "new":
        checkpoint = init_model(RANDOM, args.model, args.seed)
        path = save_checkpoint(args.path, checkpoint.model, checkpoint.history)
        print(f"Saved {path}")
    else:
        checkpoint = load_checkpoint(args.path)
    print(describe(checkpoint))


if __name__ == "__main__":
    main()
