# Epoch

Two bases at the ends of one lane: train an army, hold the line, and race your
enemy through the ages. The rules are in [RULES.md](RULES.md). Epoch has no
window of its own; people play it on the Arena website.

| File | What it holds |
| --- | --- |
| `epoch/rules.py` | Every number: ages, units, turrets, the economy and the lanes. Balancing is a change here |
| `epoch/battle.py` | A match, tick by tick, and what a bot observes |
| `ais/` | The scripted bots, one file each |
| `arena_game.py` | The adapter the platform plays Epoch through |

From the repository root:

```sh
python -m arena match epoch --map meadow --bots balanced_ai counter_ai --replay match.json
python -m arena tournament epoch
```

## Writing a bot

Add `ais/<name>_ai.py` with a `choose_action(observation)` function that returns
one of `epoch.ACTIONS` (`"melee"`, `"ranged"`, `"heavy"`, `"age"`, `"turret"`,
`"sell"`) or `None` to wait. Its docstring's first paragraph describes it on the
site.

The observation (`epoch.Observation`) is the whole match seen from the bot's
own end of the lane: `me` and `enemy` hold each side's age, gold, XP, base hit
points, queue, turrets and units, front unit first, with each unit's
`distance` from the bot's end. `observation.can(action)` and
`observation.price(action)` say what an action would cost and whether it
would be refused, `spare` is the gold the bot can spend without putting off its
next age, and `threat` is how far the nearest enemy is from the bot's base.
`ais/balanced_ai.py` is a short example.
