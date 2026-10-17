# Rules v2

A match has two tanks, each starting with three health and 100 energy. The deterministic simulation runs at 30 ticks/second and ends after 90 seconds. Elimination determines the winner; simultaneous destruction or the clock expiring is a draw. All are terminal game outcomes.

Each tick: regenerate energy and prepare shields/aim; propose both tanks' movement; resolve their overlap; fire; sweep projectiles; apply simultaneous damage; determine the result. Inputs are collected from the same pre-tick state. The order of keys in the action dictionary is irrelevant.

## Energy and shielding

- Regeneration: 12 energy/second.
- Full-speed movement: 16 energy/second, proportional to requested movement magnitude. Insufficient energy reduces movement. Holding movement can consume all regeneration before enough energy accumulates to shoot.
- Firing: 32 energy, 0.5-second cooldown. Aim changes immediately; aim zero preserves the existing direction.
- Shield activation: 16 energy plus prepaid minimum drain. At 30 Hz the minimum is six ticks (0.20 seconds), costing another 7.2 energy. Activation therefore requires and consumes **23.2 energy**.
- The six prepaid ticks include the activation tick. Releasing early does not cancel them or refund energy. `shield_ticks_remaining` counts prepaid future ticks.
- After commitment, holding costs 1.2 energy/tick. Release or insufficient energy ends the shield. Reactivation pays again. No extra activation delay or cooldown exists.
- Shielding blocks shots in every direction, prevents firing and reduces movement speed to 60%. Regeneration continues.

`RulesConfig.preset("legacy_shield")` has no activation charge or minimum commitment beyond the current tick. It exists for controlled comparisons and still uses corrected v2 physics.

## Geometry

Tanks are circles of radius 18 and projectiles have radius 4. Shots spawn at the tank center and move at 420 units/second; the visual muzzle does not move the collision origin. The projectile ignores its owner until its first bounce. It can then hit its owner, including immediately when the owner shoots into a nearby wall.

Projectile-wall sweeps use flat faces and circular corner contacts. Moving segments already overlapping a tank contact at time zero. Initial wall penetration is resolved as immediate wall contact. Projectiles bounce once; their second wall contact removes them. Tanks are tested at their post-movement positions rather than using a continuous moving-target sweep. Equal-time contacts follow deterministic ordering. Tank separation redistributes correction when a wall blocks one tank; if separation cannot be resolved, both return to their pre-move positions.

The current tank movement is axis-separated sliding. This is a deliberately simple discrete movement rule, not a rigid-body dynamics solver.

## Outcomes and faults

Three invalid actions/exceptions forfeit a controller. Import/factory/reset errors count as immediate startup failure. When both controllers fail, the match is invalid and excluded from standings, rather than counted as a competitive draw. Controllers are local trusted Python: latency is measured but no hard process timeout is imposed.

Gym terminal rewards are +1/-1/0. Damage shaping is -0.1 per learner health lost and +0.1 for actual enemy health removed by learner-owned shots. Self-hits only receive the negative term. The game clock produces `terminated=True`, `truncated=False`; an optional external `max_episode_steps` cutoff can truncate a still-running game.
