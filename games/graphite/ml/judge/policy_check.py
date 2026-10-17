"""Did teaching help? ``python -m ml policy-check --policy ml/runs/policy1/best.pt --judge ml/runs/judge1/best.pt``

Three tests on fresh evaluation maps: the taught network on instinct against
the network it was taught from; the look-ahead with the taught network
against Beast (our strongest player); and against the original Atlas. See
``ml/judge/check.py``.
"""

from ml.judge.check import policy_main as main


if __name__ == "__main__":
    main()
