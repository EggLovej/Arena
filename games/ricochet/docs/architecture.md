# Architecture

## Boundaries

- `game`: deterministic simulation, immutable public types (including action validation, event kinds and map serialization), configuration, geometry, map validation, spawn jitter and the seeded map generator (`mapgen.py`, reached through `load_map("generated:SEED")`). Standard library only; it imports nothing above it.
- `runtime`: bot loading, match lifecycle, controller faults/timing, action replays and command routing. Tournaments are the Arena's (`python -m arena tournament ricochet` from the repository root). Display/learning libraries are imported only by the commands that need them.
- `training`: encoders, Gymnasium/Parallel adapters, PPO training, checkpoint controllers and evaluation. Reward shaping and neural representation belong here.
- `display`: input bindings (`controls.py`), viewport conversion, immutable snapshot rendering, effects, local cosmetic RNG and audio. It never mutates the game engine. Rendering consumes `RenderSnapshot` and `GameEvent`; contact events carry positions/normals where relevant. Drawing styles live in `display/styles/` (see [themes](themes.md)).
- `bot`: the controller protocol, plus the optional `BaseController` and shared observation helpers for bots.
- `ais` / `maps`: contributor plugins and data. Wheels bundle fallback copies; working-directory plugin/data directories take precedence (`resources.contributor_directory`).

The core entry points are construction/reset via a new engine or `GameEngine.from_state`, `observation_for(player)`, `context_for(player)`, `step(joint_actions)`, results, `capture_state`, `restore_state` and `render_snapshot`. Actions must be validated before stepping. `runtime.match.ControllerSlot` applies validation, fault counting and timing for matches and the Gym opponent alike. Versions bind rules, encoders and replay formats to their meaning.

## Replays

A replay (schema 2) includes the full initial state/map/rules, controller code or checkpoint digest, engine version, per-step actions/events/hashes and snapshots every 300 ticks. Verification and visual playback share `replay_steps(data, start_tick)`. Seeking restores the nearest earlier checkpoint and verifies subsequent transitions. A final snapshot handles forfeits, including both-controller failures. Replays of an older schema are refused rather than played back under changed rules.

Hashing is optional for training. `capture_state` is independent of verification mode. Full state hashes include rules/map and next projectile ID; they establish same-runtime replay parity, not a guarantee of identical floating-point behavior on every platform.

## In the Arena

The Arena's platform (`platform/arena`) plays Ricochet through `arena_game.py`, which needs only `game`, `bot`, `runtime/loading.py`, `themes` and the bots: the same code runs in tournaments and, in Pyodide, on the website. The platform brings its own match runner, tournaments, ratings and replays, so `runtime/match.py` and `runtime/replay.py` serve only the standalone game: its window, its replay viewer and training.
