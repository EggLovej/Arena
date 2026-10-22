# Epoch

Two bases at the ends of one lane: train an army, hold the line, and race your
enemy through the ages, after the Flash game Age of War. The rules are in
[RULES.md](RULES.md). Epoch has no window of its own; people play it on the
Arena website.

| File | What it holds |
| --- | --- |
| `epoch/rules.py` | Every number: ages, units, turrets, specials, the economy and the lanes. Balancing is a change here |
| `epoch/battle.py` | A match, tick by tick, and what a bot observes |
| `ais/` | The scripted bots, one file each |
| `arena_game.py` | The adapter the platform plays Epoch through |

From the repository root:

```sh
python -m arena match epoch --map meadow --bots balanced_ai turtle_ai --replay match.json
python -m arena tournament epoch
```

## Writing a bot

Add `ais/<name>_ai.py` with a `choose_action(observation)` function that returns
one of `epoch.ACTIONS` or `None` to wait: `"melee"`, `"ranged"`, `"heavy"` and
`"super"` train a unit, `"turret1"` to `"turret3"` build the age's cheapest,
middle or dearest turret, `"place"` opens a turret place, and `"sell"`, `"age"`
and `"special"` do what they say. Its docstring's first paragraph describes it
on the site.

The observation (`epoch.Observation`) is the whole match seen from the bot's
own end of the lane: `me` and `enemy` hold each side's age, gold, XP, base hit
points, queue, turrets, the tick its special last struck and its units, front
unit first, with each unit's `distance` from the bot's end.
`observation.can(action)` and `observation.price(action)` say whether an action
would be refused and what it would cost, and `threat` is how far the nearest
enemy is from the bot's base. `ais/balanced_ai.py` is a short example.
