# Ricochet v2 verification — 11 October 2026

## Delivered

- Corrected projectile spawning/sweeps, initial overlap contacts, tank separation, terminal draws and self-hit shaping.
- New upfront shield economics, explicit rules version and retained legacy-shield comparison preset.
- Typed bot support, startup errors/invalid matches, immutable rules context and complete capture/restore/clone.
- Embedded-map replay v2, per-tick verification and 300-tick seeking checkpoints shared by visual/headless playback.
- Advancing reproducible episode seeds, jitter scenarios, 281-feature encoder v2 and PettingZoo Parallel adapter.
- CPU PPO with frozen opponent pools, manifests/checkpoint lineage, resume, evaluation and benchmark commands.
- Robot-sport rendering, separate HUD viewport, scaled mouse aiming, mouse menus, replay controls, effects and audio.
- Optional dependency groups and documented reusable boundaries without introducing a shared-game framework.

## Automated and packaging checks

**50 tests passed.** Coverage includes the original game behavior plus wall-adjacent shots, arena boundaries, rounded corners/tangency, initial circle contacts, shield payment/commitment/release, simultaneous outcomes, restored/clone parity, embedded-map replay and seeking, invalid matches, dataclass bots, Gym seeding/rewards/cutoffs, encoder capacities, the Parallel API checker, renderer state independence, mouse/menu/replay controls, audio failure, checkpoint version validation, opponent selection and optimizer/counter resume.

Ruff and `git diff --check` passed. Two warnings come from PettingZoo's deprecated module loader during its own API test.

Fresh core-only wheel and editable installations were tested from `/tmp`, outside the source tree. Both discovered the three maps and six scripted bots and completed a headless match. The core imported none of Pygame, NumPy, Gymnasium or Torch. Wheel contents include fallback bots/maps and the original synthesized audio assets.

The checkout's existing `.venv` repeatedly acquires macOS hidden flags on `.pth` files, which Python ignores. A non-editable install works without these files. README commands use this mode; developers must rebuild after edits. A clean editable environment outside this folder also worked. The external process responsible for setting the flags has not been identified.

## Learning smoke test

Two development smoke runs were preserved: `runs/smoke_v2` before expanded telemetry, and `runs/smoke_final` after it. Each ran 8,192 transitions and saved checkpoints at 4,096 and 8,192. Finite PPO metrics and successful checkpoint loading were observed. The final run's four-game quick validation score was **0.5**, entirely from draws. Full neural training was not launched.

A checkpoint-backed controller completed a normal recorded match against Bank Shot with no faults; its replay reproduced the final state hash. This individual win is a plumbing check, not a strength claim.

The final smoke policy's held-out Channels test ran **66 valid matches: 0 wins, 44 draws, 22 losses**, score 0.3333 (win + half-draw). It is not a strong trained policy. This test was run once for reporting, not used to select or tune a checkpoint.

The separate resume integration test trains very short rollouts and verifies preserved optimization/update counters, checkpoint versions, new-directory lineage and finite metrics. Resuming intentionally starts fresh episodes.

## Shield comparison

Both presets use the corrected physics. Ten bot pairs, three maps, two seats and two presets produced 120 games. All had zero bot faults.

| Metric | Legacy shield | New default |
|---|---:|---:|
| Games | 60 | 60 |
| Draws | 30 | 28 |
| Mean duration, seconds | 49.82 | 46.80 |
| Shots blocked | 374 | 375 |
| Reactive Defender record | 2 wins / 22 draws / 0 losses | 2 wins / 22 draws / 0 losses |
| Close Range record | 9 wins / 10 draws / 5 losses | 11 wins / 8 draws / 5 losses |

The new activation cost and minimum duration modestly changed this tournament but **did not eliminate defensive stalemates**. Preserve that limitation when interpreting training results. More aggressive strategies and additional controlled balance experiments remain useful; no automatic rule tuning was performed against this test set.

## Local performance

One 10,000-step benchmark, excluding neural inference/optimization:

| Path | Steps/second |
|---|---:|
| Engine, complete verification hashes | 18,791 |
| Engine, hashes disabled | 49,939 |
| Gym against Bank Shot | 11,884 |
| One scripted match | 16,139 |

The two engine measurements execute identical no-op traces with different verification settings. They demonstrate optional hash overhead, not general training throughput. Original review measurements used the previous rules and are not a controlled comparison to v2.

## Visual checks and limits

Pygame frames and result overlays were rendered on all three maps and inspected. Automated dummy-SDL checks exercise menu clicks, viewport-scaled aim, pause/step/seek/speed controls and one-flip-per-frame result rendering. State hashes remain unchanged by rendering. Sound assets load and the unavailable-audio path is tested; subjective sound quality and human control feel have not been evaluated in a hands-on session.

The desktop game remains a two-player, trusted-local-bot application. There is no arbitrary-code sandbox, browser port, network multiplayer, accelerated engine or claim of optimal learned play.

## Evidence

- [Shield match records](verification_data/shield_comparison.json)
- [Benchmark measurements](verification_data/benchmark_v2.json)
- [Held-out smoke evaluation](verification_data/smoke_generalization.json)
- [Smoke metrics](verification_data/smoke_metrics.jsonl)
- [Smoke manifest](verification_data/smoke_manifest.json)
- [Menu](screenshots/menu.png), [Classic](screenshots/classic.png), [Crossfire](screenshots/crossfire.png), [Channels](screenshots/channels.png)

The recorded smoke manifest describes the code at that run. Later changes add provenance fields, documentation, and rendering polish; numbered run artifacts were not retroactively edited.
