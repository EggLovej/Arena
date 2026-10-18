"""Save a model to a file and load it back.

A checkpoint is the only thing trainers hand to each other. It holds:

- which architecture it is, and the settings it was built with
- the weights
- its history: one entry per step that produced it, such as a random start,
  then imitation, then self-play
- the feature lists the encoders had when it was saved

A model only makes sense with the inputs it was trained on, so loading is
refused if the encoders have changed since. Features added at the end of a
list since then are fine: the model reads only the ones it was trained with.

From the game directory:

    .venv/bin/python -m ml checkpoint new ml/runs/demo/random.pt --seed 0
    .venv/bin/python -m ml checkpoint new ml/runs/demo/wide.pt --set width=128 heads=8
    .venv/bin/python -m ml checkpoint show ml/runs/demo/random.pt
"""

import argparse
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from io import BytesIO
from pathlib import Path

import torch

from ml.encoding.action_encoder import AMOUNT_OPTIONS
from ml.encoding.state_encoder import BASE_FEATURES, GLOBAL_FEATURES, PAIR_FEATURES, ROUTE_FEATURES
from ml.models import DEFAULT_MODEL, MODELS, PolicyValueModel, build_model, name_of


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


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu", data: bytes | None = None) -> Checkpoint:
    """The checkpoint at ``path``, or in ``data``: the bytes read from ``path`` earlier."""

    path = Path(path)
    if data is None and not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    # weights_only refuses files that contain code, so a checkpoint cannot run anything on load.
    contents = torch.load(path if data is None else BytesIO(data), map_location="cpu", weights_only=True)

    if contents.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} uses checkpoint format {contents.get('format_version')}, expected {FORMAT_VERSION}")
    model_name = contents["model_name"]
    if model_name not in MODELS:
        raise ValueError(f"{path} holds a '{model_name}' model, which is not listed in ml/models")

    current = _current_encoding()
    changed = [
        name
        for name, saved in contents["encoding"].items()
        if list(saved) != current.get(name, [])[: len(saved)]
    ]
    if changed:
        raise ValueError(
            f"{path} was trained with different {', '.join(changed)} than the encoders produce now. "
            "Retrain it, or restore the encoders it was trained with."
        )

    # A table that did not exist when the model was saved is one it never reads.
    config = dict(contents["model_config"])
    config_names = {setting.name for setting in fields(MODELS[model_name][1])}
    for table in current.keys() - contents["encoding"].keys():
        if table in config_names:
            config.setdefault(table, 0)
    weights = contents["weights"]
    # The capture output (removed 2026-10-18) never steered a move, so a network saved with one plays without it.
    if config.pop("capture_head", False):
        weights = {name: value for name, value in weights.items() if not name.startswith("capture_out.")}
    model = build_model(model_name, **config)
    model.load_state_dict(weights)
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
        "route_features": list(ROUTE_FEATURES),
        "global_features": list(GLOBAL_FEATURES),
        "amount_options": list(AMOUNT_OPTIONS),
    }


def model_settings(model_name: str, assignments: list[str]) -> dict:
    """Turn ``["width=128", "layers=4"]`` into settings of the model's config, with the right types."""

    config_fields = {setting.name: setting for setting in fields(MODELS[model_name][1])}
    settings = {}
    for assignment in assignments:
        name, _, value = assignment.partition("=")
        if name not in config_fields or not value:
            raise SystemExit(f"Unknown setting '{assignment}'. Settings of {model_name}: {', '.join(config_fields)}")
        try:
            kind = type(config_fields[name].default)
            if kind is bool:
                # bool("no") would be True: spell out what counts as yes and no.
                if value.lower() not in ("yes", "true", "1", "no", "false", "0"):
                    raise ValueError(value)
                settings[name] = value.lower() in ("yes", "true", "1")
            else:
                settings[name] = kind(value)
        except ValueError:
            raise SystemExit(f"Setting '{name}' must be like {config_fields[name].default!r}, not '{value}'")
    return settings


def parse_args():
    parser = argparse.ArgumentParser(description="Create or inspect a checkpoint.")
    commands = parser.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="Save a model with fresh random weights")
    new.add_argument("path")
    new.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    new.add_argument("--seed", type=int)
    new.add_argument(
        "--set", nargs="+", default=[], metavar="SETTING=VALUE",
        help="Change a setting of the architecture, for example width=128 layers=4",
    )

    show = commands.add_parser("show", help="Print what a checkpoint contains")
    show.add_argument("path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "new":
        checkpoint = init_model(RANDOM, args.model, args.seed, **model_settings(args.model, args.set))
        path = save_checkpoint(args.path, checkpoint.model, checkpoint.history)
        print(f"Saved {path}")
    else:
        checkpoint = load_checkpoint(args.path)
    print(describe(checkpoint))


if __name__ == "__main__":
    main()
