# Ricochet — Robot Arena

A deterministic two-player tank duel for human play, scripted bots and local reinforcement learning. Move and aim independently, manage a shared energy reserve, and use one-bounce projectiles to attack around cover.

![Neon Circuit](docs/screenshots/neon-circuit_preview.png)

## Install and play

Python 3.11+ is required. From this directory:

```bash
pip install -e ".[display]"
ricochet
```

The menu supports mouse or keyboard selection. WASD moves, mouse aims, left mouse fires, and Space/right mouse shields. F1 toggles analysis overlays, M mutes, +/- adjusts volume, and Escape returns to the menu. The scoreboard is outside the playable arena and mouse aiming respects window scaling.

Dependency extras are `display` (Pygame), `training` (NumPy/Gymnasium/PettingZoo), `learn` (PPO/PyTorch), and `dev` (pytest/Ruff). A core-only installation can run scripted headless matches without any of these libraries:

```bash
pip install -e .
ricochet match --headless --p1 direct_ai --p2 bank_shot_ai
```

## Bots, matches and recordings

Copy `ais/template_ai.py`, subclass `BaseController` and implement `act(observation)`; `create_controller()` returns an instance. The controller receives an immutable complete observation; rules are available as `self.rules`. `ricochet.bot` offers shared helpers (threats, line of sight, path finding, lead aiming). See [AI API](docs/ai_api.md).

```bash
ricochet match --headless --record replays/example.json
ricochet replay replays/example.json --headless
ricochet replay replays/example.json
```

### Generated maps

`generated:SEED` works anywhere a map name does. Each seed deterministically builds a mirror- or half-turn-symmetric arena, then shifts every free wall edge and both spawns by up to 16 px, so maps are nearly but not exactly symmetric. A map is accepted only if it is playable: walls at least 24 px thick, every gap closed or wide enough for a tank, room around the spawns, and the tanks able to reach each other. `GeneratorConfig` in `ricochet/game/mapgen.py` tunes the generator.

```bash
ricochet match --map generated:42 --p1 human --p2 bank_shot_ai
ricochet generate-map --seed 42 --output maps/arena_42.json  # Keep one for the menu.
```

Replay controls: Space pauses, `.` advances a tick, Left/Right seek five seconds, Up/Down change speed, Home restarts, and Escape exits. V2 replays embed the original map, rules and state. V1 recordings require the old `afc5ee0` revision because the physics changed.

## Train and evaluate

Install all extras first (`pip install -e ".[display,learn,dev]"`). Training runs require a **new output directory**:

```bash
ricochet train --output runs/my_smoke --smoke
ricochet evaluate runs/my_smoke/latest.zip --quick
ricochet match --p1 human --p2 checkpoint:runs/my_smoke/latest.zip
```

The smoke command trains for 8,192 transitions. It verifies the pipeline; it does not establish a strong policy. `runs/*/best.zip` checkpoints also appear automatically in the menu.

For a longer, explicitly launched three-seed experiment:

```bash
ricochet train --config configs/ppo_three_seeds.json --output runs/ppo_experiment
ricochet evaluate runs/ppo_experiment/seed_0/best.zip --suite test --output results/generalization.json
```

Resume into a new directory with `train --resume PATH --output NEW_DIRECTORY`. Optimizer, model and step counters persist; episodes restart. See [training and evaluation](docs/training.md) for configuration, checkpoint lineage, frozen opponents and evaluation boundaries.

## Rules, architecture and checks

- [Rules v2](docs/rules.md): shield costs, collision ordering and outcomes.
- [Architecture](docs/architecture.md): module boundaries, reusable contracts and state capture.

```bash
pytest
ruff check .
```

Bots and checkpoints are trusted local code. This release has no network multiplayer or public upload service.

### Visual themes

Neon Circuit is the default. Use the menu's **Theme** selector to switch between Neon Circuit and Robot League. Matches and replay playback also accept `--theme neon-circuit` or `--theme robot-league`. See [theme architecture and extension guide](docs/themes.md).

### Strong scripted opponent

Select **Sentinel** in the menu, or run `ricochet match --p1 human --p2 sentinel_ai`. It uses analytical interception, validated ricochet shots, obstacle navigation and predictive dodging, with shields disabled. See [design and evaluation](docs/sentinel.md).
