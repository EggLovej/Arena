# Arena

A website of games that classic (non-generative) AI bots play: scripted
strategies, search, and neural networks trained by self-play. Watch the bots
play each other and see who leads the ratings.

The first game is **Graphite**, real-time strategy on a graph.

## What's where

| Folder | What it holds |
| --- | --- |
| `games/<game>/` | One game: its rules engine, maps, bots (`ais/`), training code (`ml/`) and `arena_game.py`, the adapter the platform talks to |
| `games/<game>/checkpoints/` | The one network each neural bot plays (training output in `ml/runs/` stays out of git) |
| `platform/arena/` | What every game shares: the game contract, match runner, replays, tournaments, ratings, and the files the website reads |
| `platform/tests/` | Tests for the platform, with a tiny Race game that is also a template for new games |
| `web/` | The website (Next.js). Vercel deploys this folder only |
| `web/public/data/` | The published tournaments: leaderboards, match lists and replays, written by `arena tournament` |
| `.github/workflows/` | CI, and the Tournament workflow that plays the leaderboard matches and commits the results |

## Running it

Python 3.12 or newer, and Node 20.9 or newer.

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
back exactly. It does nothing if the bots, maps, rules and shared match code
are unchanged since the published one (`--force` plays it anyway). A full
Graphite tournament is about 700 matches; most take under a second, but the
search bots take up to a few minutes a match, so it runs for an hour or two
on four cores.

Graphite itself still runs on its own, with its Pygame client and training
code: see [games/graphite/README.md](games/graphite/README.md).

## Ratings

Every match is a ranking of its players. The ratings are the Plackett-Luce
model fitted to all of a tournament's results at once: with two players this
is the Elo model, and it extends to any number of players per match. They are
shown on the Elo scale, where 100 points more wins about 64% of games, with
one bot pinned (Graphite pins Strategic AI at 1000). Each match size gets its
own table, since being good in a duel and in a three-way fight are different
skills. The ± is a 95% interval for a bot's rating relative to the others.
See `platform/arena/ratings.py`.

## The Tournament workflow

`.github/workflows/tournament.yml` runs on every push to `main` that touches
`games/` or `platform/arena/`, every Monday, and on demand (Actions >
Tournament > Run workflow, with "force" to replay unchanged games). It plays
each game's tournament, refreshes the tables, and commits `web/public/data`
to `main`, which Vercel then deploys. It needs to push to `main`: if `main`
gets branch protection, allow GitHub Actions to bypass it.

## Deploying

On Vercel, import this repository and set **Root Directory** to `web`. The
framework (Next.js) is detected; no environment variables are needed. Every
page is static, built from `web/public/data`.

## Adding a bot to Graphite

Add `games/graphite/ais/<name>.py` with a `choose_action(state)` function (see
`ais/careful_ai.py`), or a `create_bot()` returning an object with
`reset(context)` and `act(state)`. Its docstring's first paragraph describes it
on the site unless `BLURBS` in `arena_game.py` has a line for it. A neural bot
also commits the one checkpoint it plays to `checkpoints/`. Open a pull
request; once merged, the Tournament workflow rates it.

## Adding a game

1. Put the game in `games/<id>/`, with its rules separate from any user interface.
2. Write `games/<id>/arena_game.py` defining `GAME`, following `platform/arena/contract.py`.
   `platform/tests/fixtures/games/race/arena_game.py` is a complete small example,
   and `games/graphite/arena_game.py` a real one.
3. Add a board for it to the website: `web/src/games/<id>/` with a component that
   draws one snapshot, registered in `web/src/games/index.ts`. The replay player,
   match lists, leaderboards and bot pages are shared.
4. Run `python -m arena tournament <id>` and commit `web/public/data`, or let the
   Tournament workflow do it after merging.
