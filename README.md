# Arena

A website of games that classic (non-generative) AI bots play: scripted
strategies, search, and neural networks trained by self-play. Watch the bots
play each other, see who leads the ratings, and play against them yourself.

The games are **Graphite**, real-time strategy on a graph, **Ricochet**, a
tank duel where every shot may bounce once, and **Epoch**, two bases on one
lane racing each other through Age of War's five ages. Epoch is hidden from
the site for now (see `web/src/games/index.ts`); its tournament still runs.

## What's where

| Folder | What it holds |
| --- | --- |
| `games/<game>/` | One game: its rules engine, maps, bots (`ais/`), training code, and `arena_game.py`, the adapter the platform talks to, with `files.json`, the files its matches load |
| `games/<game>/checkpoints/` | The one network each neural bot plays (training output stays out of git) |
| `platform/arena/` | What every game shares: the game contract, match runner, replays, tournaments, ratings, and the files the website reads |
| `platform/tests/` | Tests for the platform, with a tiny Race game that is also a template for new games |
| `web/` | The website (Next.js). Vercel deploys this folder; its build also copies the games' Python for the play page |
| `web/public/data/` | The published tournaments: leaderboards, match lists and replays, written by `arena tournament` |
| `.github/workflows/` | CI, and the Tournament workflow that plays the leaderboard matches and commits the results |

## Running it

Python 3.12 or newer, and Node 22.6 or newer.

```sh
python -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # or plain pip on a Mac
pip install -e ".[neural,test]"

python -m arena games                                   # games, maps and bots
python -m arena match graphite --map classic --bots strategic_ai careful_ai --replay match.json
python -m arena verify match.json                       # replays its actions and compares
python -m arena tournament graphite --workers 4         # every pairing, rated, into web/public/data
python -m arena publish graphite                        # redo game.json and the tables without playing

python -m pytest                                        # platform tests

cd web && npm install && npm run dev                    # the site, at http://localhost:3000
```

A tournament plays every group of bots on every map in every seating, keeps
one replay of each group on each map, and checks that every kept replay plays
back exactly. It does nothing if the files its matches load (the game's
`arena_game.py` and `files.json`), the shared match code and the pinned
packages are unchanged since the published one (`--force` plays it anyway). CI
plays the published replays back too; one recorded before a change to the
game's rules, maps or adapter may no longer match, which is only noted, since
the next tournament replaces it. A full Graphite tournament is about 700
matches; most take under a second, but the search bots take up to a few
minutes a match, so it runs for an hour or two on four cores. Ricochet's 240
matches take a few minutes, and Epoch's 60 under a minute.

Graphite and Ricochet still run on their own, with their Pygame clients and
training code: see [games/graphite/README.md](games/graphite/README.md) and
[games/ricochet/README.md](games/ricochet/README.md). Epoch is played on the
site only: see [games/epoch/README.md](games/epoch/README.md).

## Ratings

Every match is a ranking of its players. The ratings are the Plackett-Luce
model fitted to all of a tournament's results at once: with two players this
is the Elo model, and it extends to any number of players per match. They are
shown on the Elo scale, where 100 points more wins about 64% of games, with
one bot pinned (Graphite pins Strategic at 1000, Ricochet pins Direct and
Epoch pins Balanced). Each match size gets its
own table, since being good in a duel and in a three-way fight are different
skills. The ± is a 95% interval for a bot's rating relative to the others.
See `platform/arena/ratings.py`.

## The Tournament workflow

`.github/workflows/tournament.yml` runs on every push to `main` that touches
`games/`, `platform/arena/`, `pyproject.toml` or the workflow itself, every
Monday, and on demand (Actions > Tournament > Run workflow, with "force" to
replay unchanged games); a run waits for one already playing, then plays the
latest `main`. It plays each game's tournament, then takes the latest `main`
again, puts in the results it played, publishes every game's files from there
and commits `web/public/data` to `main`, which Vercel then deploys. It needs to push to `main`: if `main` gets
branch protection, allow GitHub Actions to bypass it.

## Playing against the bots

A game's Play page runs the match in the visitor's browser: the real Python
rules and bots, in [Pyodide](https://pyodide.org) (Python compiled to
WebAssembly) inside a web worker, so nothing runs on a server.
`platform/arena/live.py` plays the match a tick at a time; the page draws it
and sends the person's clicks and keys, which the game's `person(seat)` (in its
`arena_game.py`) turns into actions. As in the tournaments, every bot moves
every tick, so a bot that thinks slowly slows the game down rather than
playing worse.
The same page can seat a bot in every place, to watch any bots play each
other live. The match fills the screen, a phone's too: a game can turn a wide
board a quarter round to fill an upright phone (`PlayView.turn`; Graphite
does), and full screen (the F key) hides the rest of the page. A game may bring
music that loops while a match plays (`PlayView.music`; Epoch does), with a
button to mute it. A bot that sits out a
group of maps in the tournaments (`BotInfo.sits_out`, such as Graphite's large
maps) isn't offered for them there either, and each map keeps its own time
limit (`MapInfo.max_ticks`).
Before each build, `web/scripts/bundle-python.mjs` copies the files the game's
`files.json` lists into `web/public/play/` (not committed), and a test checks
that every file a match imports, with or without PyTorch, is listed.

Neural bots play without PyTorch there: `games/graphite/ml/play/numpy_model.py`
reads the checkpoint and runs the same network in numpy, and the tests check
it gives the same moves. A bot that needs more than its own code (numpy, a
checkpoint) says so through the game's `web_needs(bot_id)`.

Pyodide itself (about 15 MB the first time, then cached) comes from jsDelivr;
set `NEXT_PUBLIC_PYODIDE_URL` at build time to serve it from elsewhere.

## Deploying

On Vercel, import this repository and set **Root Directory** to `web`. The
framework (Next.js) is detected; no environment variables are needed. Every
page is static, built from `web/public/data`. The build also reads `games/`
and `platform/`, so keep **Include source files outside of the Root Directory
in the Build Step** on (the default).

## Adding a bot to Graphite

Add `games/graphite/ais/<name>.py` with a `choose_action(state)` function (see
`ais/careful_ai.py`), which the game's own window plays. It may also define a
`create_bot()` returning an object with `reset(context)` and `act(state)`,
which the Arena then plays instead (`ais/neural_ai.py` does, to draw its moves
from the match's seed). Its docstring's first paragraph describes
it on the site unless `BLURBS` in `arena_game.py` has a line for it. A bot may
think for up to two minutes on a move and half an hour over a match
(`move_time_limit` and `match_time_limit` in the game's `GameInfo`); one that
runs out of time, or fails to load, forfeits the match. A neural bot also
commits the one checkpoint it plays to `checkpoints/`, and lists it and any other
files it loads in `files.json`. Open a pull request; once merged, the Tournament
workflow rates it.

## Adding a board theme

Each game's pages have a theme picker, which redraws every board of that game
and is remembered on the device. Graphite's themes are the JSON files in
`games/graphite/themes/` (colours, and optionally a `texture` image or a
`grid`): add one beside `slate.json`. Ricochet's are the game's own, in
`ricochet.themes` (see `games/ricochet/docs/themes.md`). Epoch has one so
far, in `themes()` in `games/epoch/arena_game.py`. Then run
`python -m arena publish <game>` to list it in the game's `game.json`.

## Adding a game

1. Put the game in `games/<id>/`, with its rules separate from any user interface.
2. Write `games/<id>/arena_game.py` defining `GAME`, and beside it `RULES.md` and
   `files.json`, following `platform/arena/contract.py`. `platform/tests/fixtures/games/race/`
   is a complete small example, and `games/ricochet/` a real one. `platform/tests/test_games.py`
   then checks the game the way it checks every other.
3. Add a board for it to the website: `web/src/games/<id>/` with a component that
   draws one snapshot, registered in `web/src/games/index.ts` (see
   `web/src/games/types.ts`), plus a genre for the game list. The home page's
   game list and preview, the game's own pages, replay player, match lists,
   leaderboards and bot pages are shared; a game can give its own pages a look
   (colours, heading font, backdrop) as `web/src/games/graphite/look.ts` does.
   The game list follows the order in `web/src/games/index.ts`. To let people
   play it, give the view a `play` entry with the controls that lie over the
   board (Graphite's clicks are in `web/src/games/graphite/GraphiteControls.tsx`,
   Ricochet's keyboard and mouse in `web/src/games/ricochet/RicochetControls.tsx`).
   They send commands that a `person(seat)` on `GAME` turns into actions.
   The board draws each of the game's themes (`themes()` on `GAME`, the default
   first), and people pick one on the game's pages.
4. Run `python -m arena tournament <id>` and commit `web/public/data`, or let the
   Tournament workflow do it after merging.
