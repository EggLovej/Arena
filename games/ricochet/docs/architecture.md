# Architecture and reuse

## Boundaries

- `game`: deterministic simulation, immutable public types (including action validation, event kinds and map serialization), configuration, geometry, map validation, spawn jitter and the seeded map generator (`mapgen.py`, reached through `load_map("generated:SEED")`). Standard library only; it imports nothing above it.
- `runtime`: bot loading, match lifecycle, controller faults/timing, action replays and command routing. Tournaments are the Arena's (`python -m arena tournament ricochet` from the repository root). Display/learning libraries are imported only by the commands that need them.
- `training`: encoders, Gymnasium/Parallel adapters, PPO training, checkpoint controllers and evaluation. Reward shaping and neural representation belong here.
- `display`: input bindings (`controls.py`), viewport conversion, immutable snapshot rendering, effects, local cosmetic RNG and audio. It never mutates the game engine. Rendering consumes `RenderSnapshot` and `GameEvent`; contact events carry positions/normals where relevant. Drawing styles live in `display/styles/` (see [themes](themes.md)).
- `api` / `bot`: the controller protocol, plus the optional `BaseController` and shared observation helpers for bots.
- `ais` / `maps`: contributor plugins and data. Wheels bundle fallback copies; working-directory plugin/data directories take precedence (`resources.contributor_directory`).

The core entry points are construction/reset via a new engine or `GameEngine.from_state`, `observation_for(player)`, `context_for(player)`, `step(joint_actions)`, results, `capture_state`, `restore_state`, `clone`, and `render_snapshot`. Actions must be validated before stepping. `runtime.match.ControllerSlot` applies validation, fault counting and timing for matches and the Gym opponent alike. Versions bind rules, encoders and replay formats to their meaning.

## Replays

V2 includes full initial state/map/rules, controller code or checkpoint digest, engine version, per-step actions/events/hashes and snapshots every 300 ticks. Verification and visual playback share `replay_steps(data, start_tick)`. Seeking restores the nearest earlier checkpoint and verifies subsequent transitions. A final snapshot handles forfeits, including both-controller failures. V1 requires the pre-v2 revision `afc5ee0`; changing the rules is not presented as backward-compatible playback.

Hashing is optional for training. `capture_state` is independent of verification mode. Full state hashes include rules/map and next projectile ID; they establish same-runtime replay parity, not a guarantee of identical floating-point behavior on every platform.

## Reuse in the next game

Reuse the directory pattern now. Extract common code only when a second game needs it:

| Shared candidate | Game-owned behavior |
|---|---|
| Controller factory/reset/act lifecycle | Observation and action types |
| Match scheduling, budget/fault reporting | Simultaneous versus turn-based stepping |
| Tournament outputs and standings machinery | Player/team structure and outcomes |
| Replay envelope and artifact storage | State/action serialization and engine version |
| Run manifests and checkpoint lineage | Rewards, model inputs and scenario generation |
| Viewport/menu/replay widgets | Arena rendering and effects |

The current implementation is explicitly a two-player duel. It does not pretend that every future game has health, a 30 Hz clock, circles, or six neural outputs. There is no new generic inheritance framework or browser/public-upload infrastructure.
