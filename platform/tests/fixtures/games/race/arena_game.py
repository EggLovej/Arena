"""Race: the smallest game that follows the Arena contract, used by the platform's tests.

Every tick each runner moves 1, 2 or 3 steps, but a gust of wind (the same for a
given seed) holds one of them still. The first to reach the goal wins; runners
level at the goal share first place. It is also a template for a new game's
``arena_game.py``: rules, a few bots, and the adapter methods, with ``RULES.md`` and
``files.json`` beside it (everything Race's matches load is in this file).
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass

from arena.contract import BotContext, BotInfo, GameInfo, MapInfo, Outcome, SeatInfo

INFO = GameInfo(
    id="race",
    name="Race",
    tagline="Run to the goal.",
    seats=(SeatInfo("red", "Red", "#bb3933"), SeatInfo("blue", "Blue", "#2c5ba6"), SeatInfo("green", "Green", "#36845b")),
    tick_seconds=1.0,
    max_ticks=40,
    rating_anchor=("steady", 1000.0),
)


@dataclass(frozen=True)
class Track:
    info: MapInfo
    goal: int


MAPS = {
    "short": Track(MapInfo("short", "Short track", ("red", "blue"), "Ten steps."), 10),
    "long": Track(MapInfo("long", "Long track", ("red", "blue", "green"), "Twenty steps."), 20),
}


class Steady:
    def reset(self, context: BotContext) -> None:
        pass

    def act(self, observation):
        return 2


class Sprinter(Steady):
    def act(self, observation):
        return 3


class Dice:
    def reset(self, context: BotContext) -> None:
        self.random = random.Random(context.seed)

    def act(self, observation):
        return self.random.randint(1, 3)


class Broken(Steady):
    def act(self, observation):
        raise RuntimeError("this bot always fails")


class Cheater(Steady):
    def act(self, observation):
        return 7


class Garbage(Steady):
    def act(self, observation):
        return object()


class Sleeper(Steady):
    def act(self, observation):
        try:
            time.sleep(0.1)
        except Exception:  # running out of time is not an Exception, so this does not stop it
            pass
        return 2


BOTS = {
    "steady": (BotInfo("steady", "Steady", "Always two steps."), Steady),
    "sprinter": (BotInfo("sprinter", "Sprinter", "Always three steps."), Sprinter),
    "dice": (BotInfo("dice", "Dice", "One to three steps at random.", seat_counts=(2,)), Dice),
    "broken": (BotInfo("broken", "Broken", leaderboard=False), Broken),
    "cheater": (BotInfo("cheater", "Cheater", leaderboard=False), Cheater),
    "garbage": (BotInfo("garbage", "Garbage", leaderboard=False), Garbage),
    "sleeper": (BotInfo("sleeper", "Sleeper", leaderboard=False), Sleeper),
    "missing": (BotInfo("missing", "Missing", leaderboard=False), None),  # its code fails to load
}


class Runner:
    """A person on the website: runs at the speed they last chose (1, 2 or 3 steps a tick)."""

    def __init__(self):
        self.speed = None

    def command(self, data):
        if not isinstance(data, int) or isinstance(data, bool):
            raise ValueError(f"{data!r} is not a number of steps")
        self.speed = data

    def act(self, observation):
        return self.speed

    def view(self):
        return {"speed": self.speed}


class RaceMatch:
    def __init__(self, map_id: str, seed: int):
        track = MAPS[map_id]
        info, self.goal = track.info, track.goal
        self.map_id = map_id
        self.seed = seed
        self.seats = info.seats
        self.positions = {seat: 0 for seat in self.seats}
        self.tick = 0

    def acting(self):
        return self.seats

    def observe(self, seat):
        return {"you": seat, "positions": dict(self.positions), "goal": self.goal}

    def _gust(self) -> str:
        digest = hashlib.sha256(f"{self.seed}:{self.tick}".encode()).digest()
        return self.seats[digest[0] % len(self.seats)]

    def step(self, actions):
        refused = {}
        held = self._gust()
        for seat in self.seats:
            action = actions[seat]  # the runner names every acting seat
            if action is None:
                continue
            if action not in (1, 2, 3):
                refused[seat] = f"{action!r} is not 1, 2 or 3"
            elif seat != held:
                self.positions[seat] += action
        self.tick += 1
        return refused

    def finished(self):
        return max(self.positions.values()) >= self.goal

    def outcome(self):
        levels = sorted(set(self.positions.values()), reverse=True)
        ranking = tuple(tuple(seat for seat in self.seats if self.positions[seat] == level) for level in levels)
        return Outcome(ranking, "finish" if self.finished() else "time", self.tick)

    def board(self):
        return {"goal": self.goal, "seats": list(self.seats)}

    def snapshot(self):
        return {"runners": {seat: [position] for seat, position in self.positions.items()}}

    def save_state(self):
        return {"seed": self.seed, "tick": self.tick, "positions": self.positions}


class RaceGame:
    info = INFO

    def maps(self):
        return tuple(track.info for track in MAPS.values())

    def bots(self):
        return tuple(info for info, _ in BOTS.values())

    def load_bot(self, bot_id):
        bot_class = BOTS[bot_id][1]
        if bot_class is None:
            raise ModuleNotFoundError("No module named 'missing_package'")
        return bot_class()

    def new_match(self, map_id, seed):
        return RaceMatch(map_id, seed)

    def action_to_json(self, action):
        if not isinstance(action, int):
            raise TypeError(f"{action!r} is not a number of steps")
        return action

    def action_from_json(self, data):
        return data

    def person(self, seat):
        return Runner()

    def theme(self):
        return {}


GAME = RaceGame()
