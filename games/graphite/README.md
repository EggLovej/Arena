# Graphite

A real-time strategy game on a paper map, built to be played by AIs. The game
rules are separate from Pygame, so the same match can run visually or
headlessly.

## Set up

Create the virtual environment in this directory once:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt
```

## Start the game

From this directory, launch the home screen:

```bash
.venv/bin/python main.py
```

The home screen discovers the files in `maps/` and `ais/` automatically and has
two steps:

1. **Choose a map.** Every map is a tile with a small picture of its layout.
   Hovering a tile (or moving with the arrow keys) shows a large preview and its
   details; click it or press Enter to choose it. The **Random** tile picks one
   for you. Escape quits.
2. **Choose players.** Click a colour card, then click the portrait of the human
   or AI that should play it; the next card is then selected automatically.
   Coloured tokens on a portrait show which colours picked it. Three-player maps
   add a Green card. Only one colour can be Human player. Tab switches card,
   Escape goes back to the maps, and Enter or **Start match** begins.

Every AI gets a portrait generated from its file name (`avatars.py`). Words such
as `aggressive`, `careful`, `strategic`, or `neural` choose a matching look; any
other name gets its own stable combination of features. Two- and three-player
AI matches are also supported; multiple human players are intentionally not
implemented.

After a match ends, choose **Home screen** to configure another match or
**Quit** to close the game. Enter or `H` also returns home; Escape or `Q` quits.

For direct command-line or automated runs:

```bash
.venv/bin/python main.py --map classic
.venv/bin/python main.py --map ring --speed 4
.venv/bin/python main.py --map ring --headless
.venv/bin/python main.py --map triangle --green-ai strategic_ai
.venv/bin/python main.py --map three_fronts --human green
```

Map names come from JSON files in `maps/`. AI names come from Python files in
`ais/`.

To test one AI across every installed map and from every available player seat:

```bash
.venv/bin/python evaluate.py --candidate aggressive_ai --opponent careful_ai
```

The evaluator discovers maps and their active players dynamically. On a
three-player map, the candidate takes each of the three seats while the chosen
opponent controls both remaining players.

## Play against an AI

Choose your color and the AI controlling the other color:

```bash
.venv/bin/python main.py --map classic --human red --blue-ai careful_ai
.venv/bin/python main.py --map ring --human blue --red-ai aggressive_ai
```

Left-click one of your bases, then left-click a connected target. This sends
half of the source's available units. Hold Shift while clicking the target to
send all of them instead. The selected base and its edges are shown in yellow.
Left-click the selected base again, right-click, or click empty space to cancel
the selection. Press Escape to leave the match.

## Write an AI

Create a file such as `ais/my_ai.py` with one function:

```python
from game_types import SendAction


def choose_action(state):
    for source in state.my_bases():
        for target in state.neighbors(source.id):
            units = int(source.units / 2)
            if target.owner != state.player_id and units > 0:
                return SendAction(source.id, target.id, units)
    return None
```

Run it with `--red-ai my_ai` or `--blue-ai my_ai`.

`strategic_ai.py` is the strongest bundled example. It projects forces at their
arrival time, defends threatened bases, chooses efficient captures, and moves
surplus units through friendly territory toward useful frontier bases. Like the
other examples, it only uses the public `GameView` interface and contains no
map-specific names or coordinates.

`state` is an immutable, fully observable `GameView`. It contains the acting
player's ID, the current tick and time, every base, every edge, and every fleet.
An AI can return one `SendAction` per tick or `None` to wait. It receives
arbitrary base IDs and is not told the map filename. A robust AI should reason
from the graph and live state rather than recognizing coordinates or IDs.

The evaluator should use additional unseen maps. That is what makes a general
strategy necessary; an interface alone cannot prevent a submitted AI from
hard-coding the public examples.

## Tick rules

Every match uses fixed-duration ticks:

1. Every controller observes the same pre-tick state.
2. All actions are validated and valid fleets depart simultaneously.
3. Owned bases produce units.
4. Fleets move.
5. All fleets arriving at one base are resolved together.

At a contested base, units from each owner are pooled. The largest force keeps
the difference between it and the second-largest force. A tie neutralizes the
base at zero units. This rule makes simultaneous arrivals independent of Python
list order.

## Add a map

Copy one of the files in `maps/` and give every base an arbitrary unique ID.
Maps are checked when loaded: owners, coordinates, non-negative values, edge
endpoints, duplicate edges, positive travel times, required starting players,
and graph connectivity must all be valid. Owners use `RED`, `BLUE`, and
optionally `GREEN` from `game_types.py`; maps without Green remain two-player.

Two optional fields: `tagline`, a few words on the map's character, and
`time_limit`, the seconds of game time before a match is a draw when the map
needs longer than the usual three minutes. The large maps (Clockwork Citadel,
Shattered Spiral, Tectonic Archipelago and Threefold Crucible, each over 100
bases) use ten minutes. On maps with 80 or more bases, bases are drawn smaller
and show their production badge and name only while hovered or selected, and a
window too big for the screen is shrunk to fit.

## Files

- `game_types.py`: the public AI contract and map data types
- `game.py`: mutable match state and deterministic rules
- `loading.py`: dynamic map and AI loading plus map validation
- `renderer.py`: read-only Pygame visualization
- `sizing.py`: production-scaled base sizes shared by drawing and click detection
- `home_screen.py`: two-step map and player selection
- `avatars.py`: portraits for AIs, generated from their names
- `theme.py`: shared atlas colors, fonts, and image loading
- `assets/`: paper texture and faction brush marks
- `human_controller.py`: mouse input translated into ordinary game actions
- `main.py`: command-line match runner
- `evaluate.py`: all-map, both-colors evaluator
- `ais/`: one-file agents
- `maps/`: dynamically selectable JSON maps
- `tests/`: rule, loading, generality, and replay tests
- `ml/`: the learned agent, its training and evaluation (see `ml/README.md`)

The visual system follows the warm Field Notes Atlas direction. Base counts,
ownership, production, route times, selection, and moving fleets are all state-
driven. A base's size grows with its production rate relative to the richest
base on the map, and the exact rate sits in a badge on its rim. Decorative dot
rows are intentionally omitted because they do not represent game data.

For an adversarial benchmark, run untrusted AI code in a separate process with
a per-move time and memory limit. This version establishes the stable interface
but intentionally does not pretend that importing a Python file is a security
sandbox.
