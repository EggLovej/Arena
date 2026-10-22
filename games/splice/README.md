# Splice

Turn-based territory on a board of hexagons: grow into a neighbouring cell or
jump two away, and every enemy piece next to where you land becomes yours. The
rules are in [RULES.md](RULES.md). Splice has no window of its own; people play
it on the Arena website.

| File | What it holds |
| --- | --- |
| `splice/rules.py` | The board, positions and moves, with sets of cells as bit masks so bots can search fast |
| `splice/maps.py` | The maps, drawn as text |
| `splice/search.py` | What the bots share: how much a move gains, a player's lead, and alpha-beta search |
| `ais/` | The bots, one file each |
| `themes/` | The looks the website draws the board in; the first, Petri Dish, gives the seats their colours |
| `arena_game.py` | The adapter the platform plays Splice through |

From the repository root:

```sh
python -m arena match splice --map honeycomb --bots minimax_ai mcts_ai --replay match.json
python -m arena tournament splice
```

## The bots

They form a ladder from simple to strong, each a classic way to play a board game:

| Bot | How it plays |
| --- | --- |
| Random | Any legal move |
| Greedy | The move that gains most now; the leaderboard's fixed point, rated 1000 |
| Careful | Alpha-beta search two moves deep: its move and the next player's reply |
| Minimax | Alpha-beta search four moves deep |
| Monte Carlo | Monte Carlo tree search over the twelve moves that gain most in each position, 400 lines a move |

With three players the search bots are paranoid: they assume both others play
against them.

## Writing a bot

Add `ais/<name>_ai.py` with a `choose_action(position)` function, or a
`create_bot()` that returns an object with `reset(context)` and
`act(position)`. It is asked only on its own turn, with a `splice.Position`:
`position.moves()` lists the legal moves as `(from, to)` cell numbers and
`position.play(move)` gives the position after one, so a bot can look ahead.
`position.pieces` holds each player's cells as a bit mask, in turn order, and
`position.turn` is the bot's own number in that order. Draw any randomness
from `random.Random(context.seed)`, so a seeded match plays the same again.
`ais/greedy_ai.py` is a short example. Its docstring's first paragraph
describes it on the site.
