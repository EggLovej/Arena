"""Arena: the code every game on the site shares.

- ``contract``: what a game and a bot look like to everything else
- ``registry``: finding the games in ``games/``
- ``runner``: playing one match and recording its replay
- ``frames``: storing a snapshot of every tick compactly
- ``replay``: reading replays and checking that they play back exactly
- ``ratings``: ratings from results, for any number of players per match
- ``tournament``: who plays whom, run in parallel
- ``leaderboard``: standings, ratings and head-to-head tables from results
- ``publish``: the JSON files the website reads

Run ``python -m arena --help`` for the command line.
"""
