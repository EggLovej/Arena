# Writing a Ricochet controller

Add a public Python file under `ais/`. A separate factory-created controller serves each seat:

```python
from ricochet import BaseController, ControlAction, Vec2
from ricochet.bot import threats


class MyController(BaseController):
    def act(self, observation):
        target = observation.opponent.position - observation.me.position
        shield = any(
            (shot.position - observation.me.position).length() < 80
            for shot in threats(observation)
        )
        return ControlAction(move=Vec2(0, 0), aim=target, fire=not shield, shield=shield)


def create_controller():
    return MyController()
```

Modules using dataclasses and postponed annotations are supported. Put match state on controller instances. Factories must return an object with callable `reset` and `act` methods. `reset` receives a frozen `MatchContext` with seed, player/opponent IDs, map ID, tick rate, duration and immutable rules. Both `act` calls observe the same pre-tick state.

`BaseController` is optional: its `reset` stores `self.context`, `self.rules`, `self.player_id` and `self.opponent_id`. Override `reset` for extra per-match state and call `super().reset(context)` first. Any object with `reset` and `act` still works.

## Helpers

`ricochet.bot` contains pure functions of the observation, so every bot shares one definition of the game geometry:

- `threats(observation)`: projectiles that can hurt you — the opponent's, plus your own after they bounce.
- `on_collision_course(projectile, target, radius, seconds)`: whether a projectile's straight path passes within `radius` of a point.
- `line_of_sight(observation, start, end)` and `position_is_clear(observation, position, radius)`: wall queries.
- `find_path(observation, start, goal, radius)`: grid A* waypoints around walls.
- `lead_target(shooter, target, projectile_speed)`: first-order aim at a moving tank.

## Observations and actions

`GameObservation` contains time/arena dimensions, both tanks, every wall and all projectiles. Frozen nested values prevent accidental edits. Tank fields include position, velocity, aim, health, energy, firing cooldown, shield activity and remaining prepaid shield ticks. Projectile fields include ID, owner, position, velocity, bounce status and remaining lifetime. Coordinates start at the top-left, +X right, +Y down.

Actions contain movement XY, aim XY, fire and shield. Vectors must be finite and are limited to length one; nonzero aim is normalized by the engine. A zero aim keeps the previous aim. Boolean fire/shield flags must be actual booleans. Shield takes priority over firing and remains active for its prepaid commitment. See [rules](rules.md).

## Faults and time limits

In the Arena, a bot that raises an error, or returns anything that isn't an action, does nothing that tick and plays on. A bot may think for up to 120 seconds on one move (or on loading and `reset`) and 30 minutes over a match; one that runs out of time, or fails to load, does nothing for the rest of the match. The match report counts each bot's errors.

The standalone game is stricter about errors and has no time limit. An `act` fault emits a no-op and counts toward three-fault forfeiture; an import, factory or `reset` failure is an immediate forfeit. If both controllers fail, the match is invalid; JSON summaries include failure messages. Either way, bots are trusted Python with no security sandbox.

## Encoding and neural controllers

Encoder version **2**, action version **1**:

- Remaining-time fraction: 1 value.
- Self/opponent tank features: 20 values.
- Twelve wall slots, five values each: 60.
- Twenty-four projectile slots, eight values each: 192.
- Appended fields: two shield commitment fractions, two bounded arena dimensions, relative opponent XY and relative velocity XY: 8.

Total: **281 float32 values** in [-1, 1]. Projectiles are nearest-first with ID tie-breaking; active masks distinguish padded slots. Dimensions use width/(width+960), height/(height+640). Rules or observations exceeding the capacities raise an error. The six Box action values are movement XY, aim XY, fire and shield; positive trigger values activate the flags. This simple thresholded policy is the baseline, not a claim that the action distribution is optimal.

Load a trusted compatible checkpoint through `checkpoint:/path/to/model.zip` anywhere the CLI accepts a bot name. Its companion `.json` metadata is required. Rules/encoder/action incompatibility is rejected. The menu discovers `runs/*/best.zip` automatically.

## Search and multi-agent use

`GameEngine.capture_state()` returns a JSON-serializable complete state. `restore_state(state)` replaces the engine's state, and `GameEngine.from_state(state)` builds an independent engine from it. Snapshots include the map, rules, timers, projectile counter and outcome, so rollout bots need not manipulate private entities.

`RicochetEnv` provides single-agent Gymnasium play against a controller or controller factory. `RicochetParallelEnv` in `ricochet.training.parallel_env` provides simultaneous `player_0` / `player_1` observations and actions. Its reward is terminal win/loss/draw only; Gym's extra damage shaping is intentionally an adapter choice. Both use the same engine and versioned encoder.
