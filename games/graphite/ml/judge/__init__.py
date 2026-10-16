"""The judge loop: generate games, train a network that says who will win, test it as the look-ahead's judge.

    .venv/bin/python -m ml judge-games --name round1     generate games and record positions
    .venv/bin/python -m ml judge-train --data round1 --run judge1
    .venv/bin/python -m ml judge-check --judge ml/runs/judge1/best.pt

See ml/plan.md, "The judge: a separate value network".
"""
