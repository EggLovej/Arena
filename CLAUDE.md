# Arena

Games that classic AI bots play, the platform that runs and rates their
matches, and the website that shows them. README.md says what's where and how
to run each part.

## The standard

No bloat. Every change leaves the code well documented, minimal, modular and
correct.

- Write the least code that does the job: no dead code, no copies, no
  abstraction or option with a single user. Delete what a change leaves unused.
- What games share lives once, in `platform/arena/`, not in a copy per game.
- A docstring says what a module, class or function is for; a comment says why
  the code is as it is. Docs and READMEs match the code they describe.
- Tests check expected behaviour: what a game, bot, module or page promises,
  not its internals, shapes or counts. Fold a new case into a test that already
  sets up the same thing, or parametrize one, before writing another; delete a
  test that only restates the code.

## Conventions

- A game's `files.json` lists every file its rules and bots load. A bot that
  imports a new file adds it there, or `platform/tests/test_games.py` fails.
- Bots draw their randomness from the match's seed, so a seeded match plays the
  same moves again. A change that could alter how a bot plays proves it doesn't:
  `python -m arena tournament <game> --force --output <folder>` before and after,
  with the same results.
- Changes land as pull requests, one area each; the owner reviews and merges.

## Checks

CI runs these on every pull request; run the ones for what you changed first.

```sh
ruff check --select F .                        # unused imports and variables, undefined names
python -m pytest -q platform/tests             # from the repository root
cd games/graphite && python -m pytest -q tests ml/tests
cd games/ricochet && PYTHONPATH=src python -m pytest -q
cd web && npm test && npm run build            # the build typechecks, unused locals included
```

For a cleanup, `vulture platform games` lists code nothing calls. Most of what
it lists is called by a framework (Gym, stable-baselines3, pickle, signal
handlers) or by the website's worker, so read each one before deleting it.
