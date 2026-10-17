# Sentinel: a non-learning controller

Select **Sentinel** in the menu, or use `--p1 sentinel_ai` / `--p2 sentinel_ai`. It is a normal factory-based AI in `ais/sentinel_ai.py`, discovered by the existing tournament tools. It uses public observations and rules; no trained weights, external services, opponent names, private engine access, or runtime dependencies beyond the core game.

## How it works

- A visibility graph around inflated obstacle corners chooses routes, with grid A* as a fallback for awkward passages.
- A short-horizon movement search evaluates 16 directions plus waiting, goal movement and the previous action. Incoming projectiles are traced through their permitted bounce, including the bot's own returning shots. Candidate movement respects wall sliding and energy limits.
- Quadratic interception leads moving targets. Mirrored targets propose boundary and obstacle bank shots; the actual swept-circle path is checked before spending energy.
- It stops to regenerate firing energy instead of moving continuously. Observed enemy behavior informs when to apply contact pressure and when to leave room to dodge.
- Shields are **off by default**, including in the reported benchmarks. Rule values come from the match context, so changing shield economics does not invalidate a model or require retraining.

This is approximate receding-horizon planning, not minimax or an optimal-policy proof. Enemy velocity can change after aiming; defensive search assumes a constant candidate movement over a short horizon and cannot see an action the opponent has not yet taken. The close-contact tactic against stationary shield users depends on current discrete movement/projectile timing. New firing rules, tick rates or collision semantics should be re-evaluated.

## Reproduce evaluations

From the repository, using the current source:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_sentinel.py --suite authored --workers 4
PYTHONPATH=src .venv/bin/python scripts/evaluate_sentinel.py --suite heldout --seed 1000 --rounds 6 --workers 4
PYTHONPATH=src .venv/bin/python scripts/ablate_sentinel.py --workers 4
```

The evaluator uses the repository's `play_fixtures`, matches every other discovered bot from both seats, writes normal match JSON/CSV, and records source hashes and exact rules. The generated suite runs default, expensive-shield and disabled-shield presets. `heldout` denotes a seed batch chosen for validation; after tuning on its results it is development data and a fresh seed batch must be chosen.

For the existing full round robin:

```sh
ricochet auto-tournament --rounds 10 --seed 2000 --workers 4
```

## Final results

Against all six existing bots from both seats, Sentinel won **36/36 authored-map games**. On six fresh generated maps it scored **212 wins, 4 draws, 0 losses** across 216 games: 70/2/0 with default shields, 71/1/0 with expensive shields and 71/1/0 with shields disabled. Sentinel itself never requested a shield. Every draw was against Bank Shot.

These results establish a strong scripted baseline against the current pool, not an optimal policy. Mean action time averaged 2.65–3.84ms per match across the parallel suites, with occasional calls up to 102ms under load.
