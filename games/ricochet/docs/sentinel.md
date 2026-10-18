# Sentinel: a non-learning controller

Select **Sentinel** in the menu, or use `--p1 sentinel_ai` / `--p2 sentinel_ai`. It is a normal factory-based AI in `ais/sentinel_ai.py`, discovered like every other bot. It uses public observations and rules; no trained weights, external services, opponent names, private engine access, or runtime dependencies beyond the core game.

## How it works

- A visibility graph around inflated obstacle corners chooses routes, with grid A* as a fallback for awkward passages.
- A short-horizon movement search evaluates 16 directions plus waiting, goal movement and the previous action. Incoming projectiles are traced through their permitted bounce, including the bot's own returning shots. Candidate movement respects wall sliding and energy limits.
- Quadratic interception leads moving targets. Mirrored targets propose boundary and obstacle bank shots; the actual swept-circle path is checked before spending energy.
- It stops to regenerate firing energy instead of moving continuously. Observed enemy behavior informs when to apply contact pressure and when to leave room to dodge.
- Shields are **off by default**, including in the reported benchmarks. Rule values come from the match context, so changing shield economics does not invalidate a model or require retraining.

This is approximate receding-horizon planning, not minimax or an optimal-policy proof. Enemy velocity can change after aiming; defensive search assumes a constant candidate movement over a short horizon and cannot see an action the opponent has not yet taken. The close-contact tactic against stationary shield users depends on current discrete movement/projectile timing. New firing rules, tick rates or collision semantics should be re-evaluated.

## How strong it is

The Arena's leaderboard rates it against every other bot: `python -m arena tournament ricochet`, from the repository root, plays every pairing on every map from both seats. While it was developed, it won all 36 of its games on the three drawn maps against the six other bots, and 70 of 72 on six generated maps, drawing the other two against Bank Shot. That makes it a strong scripted baseline against the current pool, not an optimal policy.
