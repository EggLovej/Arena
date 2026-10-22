# Blast

Two to four bombers on a grid of walls and crates: drop bombs, break crates
for power-ups, and be the last one standing. The rules are in
[RULES.md](RULES.md). Blast has no window of its own; people play it on the
Arena website.

| File | What it holds |
| --- | --- |
| `blast/rules.py` | Every number (fuse, flames, steps, power-ups, when the walls close in), and the map loader. Balancing is a change here |
| `blast/field.py` | A match, tick by tick, and what a bot observes |
| `blast/plan.py` | What bots share: when each tile will burn, how to get somewhere without being caught, and `Bot` |
| `maps/` | One file a map |
| `themes/` | The board's themes; `garden.json` is the default and gives the seats their colours |
| `ais/` | The bots, one file each |
| `arena_game.py` | The adapter the platform plays Blast through |

From the repository root:

```sh
python -m arena match blast --map courtyard --bots digger_ai hermit_ai hunter_ai trapper_ai --replay match.json
python -m arena tournament blast --workers 4
```

## Writing a bot

Add `ais/<name>_ai.py` with a `choose_action(observation)` function that
returns one of `blast.ACTIONS` (`"up"`, `"down"`, `"left"`, `"right"`,
`"bomb"`) or `None` to stay. Its docstring's first paragraph describes it on
the site. The observation (`blast.Observation`) is the whole field except what
the crates hide: the bot itself (`me`), its `rivals`, the walls, crates,
bombs, flames and power-ups, and when each tile the walls have yet to reach
will close.

Most bots subclass `blast.Bot` instead and return it from `create_bot()`: it
gets out of the way of blasts by itself and calls `choose(observation, plan)`
only when it is safe to do something else. The `plan` (`blast.Plan`) knows
where the bot can get to and when, which tiles are safe for good, whether it
could get away from a bomb dropped here or elsewhere (`can_bomb`), the best
crates to dig for (`dig`) and whether a bomb would catch a rival (`traps`).
`ais/digger_ai.py` is a short example. A bot draws any randomness from
`self.chance`, which the match's seed sets.

## Adding a map

Add `maps/<id>.json` with a `name`, a `description`, the `mirror` the crates
are dealt with (`"x"`, `"y"` or `"xy"`) and the `rows`: `#` a wall, `.` a floor
tile that may start with a crate, `-` one kept clear and `1` to `4` the starts.
The walls close in on every map the same way, so its rows set how long that
takes.
