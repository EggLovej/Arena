"""Play many games at once and record them for training.

A "rollout" is a game played to collect experience. Both trainers start
here: imitation records a scripted teacher, self-play records the network.

Each seat is filled either by an AI file from ``ais/`` or by a model. Games
advance tick by tick in lockstep, and every model answers all of its games'
positions in one batch, which is far faster than asking one at a time. A
pool of worker processes can also leave the models to one server on the GPU
(``RolloutPool(inference_device="mps")``, see ml/play/inference.py).

For every seat marked ``record``, the result holds a trajectory: each
decision's encoded position, its legal choices, and the choice made. A
position where waiting is the only option is not a decision and is skipped.
A network with free amounts (ml/models/amounts.py) chooses units rather than
an amount option; its trajectory says so (``free_amounts``).

From the game directory:

    .venv/bin/python -m ml rollout --red untrained --blue strategic_ai --games 64
    .venv/bin/python -m ml rollout --red untrained --blue untrained --games 256 --workers 8
"""

import argparse
import multiprocessing
import os
import queue
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch

from game import Game
from game_types import BLUE, RED, MapDefinition, SendAction

from ml.encoding.action_encoder import (
    AMOUNT_OPTIONS,
    LIMITS,
    LegalActions,
    SendChoice,
    decode_action,
    encode_action,
    legal_actions,
)
from ml.encoding.forecast import build_timelines, ticks_until_arrival
from ml.encoding.state_encoder import (
    BASE_FEATURES,
    GLOBAL_FEATURES,
    ROUTE_FEATURES,
    EncodedState,
    build_layout,
    encode_state,
)
from ml.evaluation.evaluate import MATCH_SECONDS, outcome_for
from ml.map_generator import generate_map
from ml.models import PolicyValueModel, build_model, name_of
from ml.play.inference import BufferShape, InferenceServer, LocalInference, RemoteInference, Request
from ml.play.players import keeps_memory, load_player


WAIT = -1  # stored in place of a source, target, and amount when the choice was to wait
# How many amount options there were before "hold" and "surplus" were added (2026-10-03).
# Trajectories saved before then hold their legal choices in that layout.
OPTIONS_BEFORE_TACTICAL = 9


@dataclass(frozen=True)
class Seat:
    player: str  # an AI file in ais/, or the name of a model passed to play_games
    record: bool = False  # keep this seat's decisions for training
    # At most one send every this many ticks (10: once a second); the seat waits in between.
    # 0 or 1: no limit.
    min_send_gap: int = 0


@dataclass(frozen=True)
class Matchup:
    map: MapDefinition
    red: Seat
    blue: Seat


@dataclass
class Trajectory:
    """One seat's decisions in one game, in the order they were made."""

    seat: int
    player: str
    outcome: str  # "win", "draw", or "loss" for this seat
    base_ids: tuple[str, ...]
    pair_features: np.ndarray  # [bases, bases, pair features]; the same for every decision
    base_features: np.ndarray  # [decisions, bases, base features]
    global_features: np.ndarray  # [decisions, global features]
    # Route features only where there is a route, to save memory: entry i
    # belongs to the route from routes[i, 0] to routes[i, 1]. Read the full
    # table with ``route_table()``.
    routes: np.ndarray  # [routes, 2], the same for every decision
    route_features: np.ndarray  # [decisions, routes, route features]
    # The legal choices of each decision, packed eight to a byte to save memory.
    # Read them with ``allowed()``.
    allowed_bits: np.ndarray  # [decisions, packed bytes]
    # The choice made, as rows and an amount option (units, for a network with free amounts);
    # all three are WAIT when it waited.
    source: np.ndarray  # [decisions]
    target: np.ndarray  # [decisions]
    amount: np.ndarray  # [decisions]
    tick: np.ndarray  # [decisions], the game tick of each decision
    # What the model thought at the time. Missing for scripted players.
    log_probability: np.ndarray | None = None  # [decisions]
    value: np.ndarray | None = None  # [decisions]
    # For each send: 1 if its target was the sender's right after the fleet landed, 0 if not.
    # NaN for waiting and for a fleet still in flight when the game ended. Missing in
    # trajectories recorded before 2026-10-03.
    captured: np.ndarray | None = None  # [decisions]
    # How many amount options ``allowed_bits`` holds per route. Older trajectories hold fewer
    # than there are now; ``allowed`` and ``current_allowed_bits`` give them the current layout.
    amount_options: int = OPTIONS_BEFORE_TACTICAL
    # Added 2026-10-09, missing before. The units each send sent (WAIT when it waited), and
    # each route's LIMITS at each decision (ml/encoding/action_encoder.py): enough to rebuild
    # what every amount option would have sent (``option_units``), and the most and the
    # needed that a network with free amounts reads.
    units: np.ndarray | None = None  # [decisions]
    route_limits: np.ndarray | None = None  # [decisions, routes, LIMITS]
    # True when a network with free amounts made the choices: ``amount`` then holds units.
    free_amounts: bool = False

    def __len__(self) -> int:
        return len(self.tick)

    def allowed(self, decisions: slice | np.ndarray = slice(None)) -> np.ndarray:
        """Legal choices as [decisions, bases, bases, amount options], True where legal."""

        bases = len(self.base_ids)
        stored = (bases, bases, self.amount_options)
        bits = np.unpackbits(self.allowed_bits[decisions], axis=1, count=bases * bases * stored[2])
        allowed = bits.reshape(-1, *stored).astype(bool)
        if self.amount_options < len(AMOUNT_OPTIONS):
            # Options added since were not on offer when this was recorded.
            missing = np.zeros((len(allowed), bases, bases, len(AMOUNT_OPTIONS) - self.amount_options), dtype=bool)
            allowed = np.concatenate((allowed, missing), axis=-1)
        return allowed

    def current_allowed_bits(self, decisions: slice | np.ndarray = slice(None)) -> np.ndarray:
        """``allowed_bits`` in the current layout of amount options."""

        if self.amount_options == len(AMOUNT_OPTIONS):
            return self.allowed_bits[decisions]
        allowed = self.allowed(decisions)
        return np.packbits(allowed.reshape(len(allowed), -1), axis=1)

    def route_table(self, decisions: slice | np.ndarray = slice(None)) -> np.ndarray:
        """Route features as [decisions, bases, bases, route features], 0 where there is no route."""

        return spread_routes(self.route_features[decisions], self.routes, len(self.base_ids))

    def limit_table(self, decisions: slice | np.ndarray = slice(None)) -> np.ndarray:
        """Route LIMITS as [decisions, bases, bases, LIMITS], 0 where there is no route."""

        return spread_routes(self.route_limits[decisions], self.routes, len(self.base_ids))


def spread_routes(values: np.ndarray, routes: np.ndarray, bases: int) -> np.ndarray:
    """Per-route values [decisions, routes, k] as a full [decisions, bases, bases, k] table.

    ``routes`` is one [routes, 2] list for all decisions, or one per decision
    as [decisions, routes, 2].
    """

    table = np.zeros((len(values), bases, bases, values.shape[-1]), dtype=values.dtype)
    if routes.ndim == 2:
        table[:, routes[:, 0], routes[:, 1]] = values
    else:
        rows = np.arange(len(values))[:, None]
        table[rows, routes[..., 0], routes[..., 1]] = values
    return table


@dataclass
class GameRecord:
    matchup: Matchup
    winner: int | None
    ticks: int
    # Every move made, as (tick, seat, source ID, target ID, units). Enough to replay the game exactly.
    actions: list[tuple[int, int, str, str, int]]
    trajectories: list[Trajectory] = field(default_factory=list)


def play_games(
    matchups: list[Matchup],
    models: dict[str, PolicyValueModel] | None = None,
    parallel_games: int = 64,
    device: torch.device | str = "cpu",
    greedy: bool = False,
    decision_interval: int = 1,
    max_seconds: float = MATCH_SECONDS,
    seed: int | None = None,
    on_finished: Callable[[int, GameRecord], None] | None = None,
    inference: LocalInference | RemoteInference | None = None,
    temperatures: dict[str, float] | None = None,
) -> list[GameRecord]:
    """Play every matchup and return one record per matchup, in the same order.

    ``models`` maps a player name to a model; any other player name is loaded
    from ``ais/``. Model players only decide on every ``decision_interval``-th
    tick and wait in between. Scripted players always act every tick.
    A seat with a ``min_send_gap`` waits that many ticks after each of its
    sends: a model is not asked meanwhile, a scripted player is asked but its
    send is dropped.
    ``on_finished`` is called with a matchup's index and record as soon as its game ends.
    ``inference`` runs the models (see ml/play/inference.py); by default they
    run here, on ``device``, and ``models``, ``greedy``, ``temperatures`` and ``device``
    are its settings.
    ``temperatures`` maps a model's name to the temperature its moves are drawn at
    (see ``policy.choose``; below 1, its likelier moves more often still); others draw
    at 1. Only for playing: the recorded probabilities are always the network's own, so
    training must leave every temperature at 1.
    """

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    if inference is None:
        inference = LocalInference(models or {}, device, greedy, generator, temperatures)
    scripted = {}

    waiting = deque(enumerate(matchups))
    records: list[GameRecord | None] = [None] * len(matchups)
    # The games are split into groups that take turns: while the models decide
    # for one group, the next one is simulated. With local inference there is
    # one group, as nothing can happen while the models think.
    groups = [_GameGroup() for _ in range(inference.groups)]
    per_group = -(-min(parallel_games, max(len(matchups), 1)) // len(groups))

    while waiting or any(group.running for group in groups):
        for number, group in enumerate(groups):
            if group.ticket is not None:
                answers = inference.collect(group.ticket)
                group.ticket = None
                for (run, seat, encoded, legal), choice, log_p, value in zip(
                    group.asked, answers.choices, answers.log_probabilities, answers.values
                ):
                    action = decode_action(choice, legal, free=run.seats[seat].player in inference.free)
                    run.moves[seat] = action
                    if run.seats[seat].record:
                        units = None if action is None else action.units
                        run.record(seat, run.game.tick, encoded, legal, choice, units, log_p, value)

                still_running = []
                for run in group.running:
                    if run.advance():
                        still_running.append(run)
                    else:
                        records[run.index] = run.finish(inference.names, inference.free)
                        if on_finished is not None:
                            on_finished(run.index, records[run.index])
                group.running = still_running

            while waiting and len(group.running) < per_group:
                index, matchup = waiting.popleft()
                group.running.append(_RunningGame(index, matchup, max_seconds))
            if group.running:
                group.asked, request = _positions_to_decide(group.running, inference.names, scripted, decision_interval)
                group.ticket = inference.submit(number, request)

    return records


def _positions_to_decide(
    running: list["_RunningGame"],
    model_names: set[str],
    scripted: dict,
    decision_interval: int,
) -> tuple[list[tuple["_RunningGame", int, EncodedState, LegalActions]], Request]:
    """This tick's moves of the scripted players, and the positions where a model has to decide."""

    asked, request = [], Request([], [], [])
    for run in running:
        run.moves = {}
        # Both seats see the same position, so the snapshot and the fleet
        # timelines are built once and shared.
        red_state = run.game.view_for(RED)
        states = {RED: red_state, BLUE: replace(red_state, player_id=BLUE)}
        timelines = None
        for seat, spec in run.seats.items():
            state = states[seat]
            by_model = spec.player in model_names
            if by_model and (run.game.tick % decision_interval or not run.may_send(seat)):
                continue
            if not by_model:
                if seat not in run.players:
                    # An AI that remembers things during a match gets its own player in every game.
                    if spec.player not in scripted:
                        scripted[spec.player] = (load_player(spec.player), keeps_memory(spec.player))
                    player, per_game = scripted[spec.player]
                    run.players[seat] = load_player(spec.player) if per_game else player
                player = run.players[seat]
                run.moves[seat] = player(state)
                if not run.may_send(seat):
                    # Still asked, so an AI that remembers what it saw misses nothing. Waiting
                    # is all it may do, which is not a decision, so nothing is recorded.
                    run.moves[seat] = None
                    continue
                if not spec.record:
                    continue

            if timelines is None:
                timelines = build_timelines(state)
            legal = legal_actions(state, run.layout, timelines)
            if not legal.allowed.any():
                continue
            encoded = encode_state(state, run.layout, timelines)
            if by_model:
                asked.append((run, seat, encoded, legal))
                request.states.append(encoded)
                request.legal.append(legal)
                request.players.append(spec.player)
            else:
                choice = encode_action(run.moves[seat], legal)
                run.record(seat, run.game.tick, encoded, legal, choice, None if choice is None else run.moves[seat].units)
    return asked, request


@dataclass
class _GameGroup:
    running: list["_RunningGame"] = field(default_factory=list)
    asked: list = field(default_factory=list)  # the positions sent to the models, in order
    ticket: object = None  # the models' answer, or how to collect it


class RolloutPool:
    """Play games on several processor cores at once.

    Worker processes stay alive between calls, so a trainer can create one
    pool and reuse it every round::

        with RolloutPool(workers=8) as pool:
            records = pool.play(matchups, {"current": model})
    """

    def __init__(self, workers: int | None = None, inference_device: torch.device | str | None = None):
        # Measured on an M2 Pro: throughput stops rising after about six workers.
        self.workers = workers or max(1, min(6, (os.cpu_count() or 2) - 2))
        # With a device (e.g. "mps"), the models stay in this process and answer every
        # worker's positions in batches there (see ml/play/inference.py). Without one, each
        # worker runs its own copy of the models on its CPU core.
        self.inference_device = inference_device
        self.last_batching = None  # with a server: (batches, positions) of the last play
        self._pool = multiprocessing.get_context("spawn").Pool(self.workers, initializer=_prepare_worker)
        self._manager = None  # started on first use, for reporting finished games (see ``play``)

    def play(
        self,
        matchups: list[Matchup],
        models: dict[str, PolicyValueModel] | None = None,
        seed: int | None = None,
        on_finished: Callable[[int, list[int], int | None], None] | None = None,
        **settings,
    ) -> list[GameRecord]:
        """Same as ``play_games``, which also lists the available ``settings``.

        ``on_finished`` is called here, in this process, every time a game
        ends anywhere: with the matchup's index, the number of decisions
        recorded for each recorded seat, and the winner (None for a draw).
        That is enough to show progress game by game; the records themselves
        still come back all at once.
        """

        models = models or {}
        # Dealing matchups out like cards gives every worker a similar mix.
        hands = [hand for hand in (list(range(start, len(matchups), self.workers)) for start in range(self.workers)) if hand]
        messages = None
        if on_finished is not None:
            if self._manager is None:
                self._manager = multiprocessing.get_context("spawn").Manager()
            messages = self._manager.Queue()

        def report_finished() -> None:
            while messages is not None:
                try:
                    on_finished(*messages.get_nowait())
                except queue.Empty:
                    return

        server = None
        if self.inference_device is not None and models:
            server = InferenceServer(
                models, self.inference_device, settings.get("greedy", False), seed, settings.get("temperatures")
            )
            largest_hand = max(len(hand) for hand in hands)
            per_group = -(-min(settings.get("parallel_games", 64), largest_hand) // RemoteInference.groups)
            # Room for both seats of every game in a group.
            shape = BufferShape(rows=2 * per_group, bases=max(len(matchup.map.bases) for matchup in matchups))
            links = server.open(len(hands), shape)
            packed = {}
        else:
            # Models travel to the workers as plain weights and are rebuilt there on the CPU.
            packed = {
                name: (name_of(model), asdict(model.config), {k: v.cpu() for k, v in model.state_dict().items()})
                for name, model in models.items()
            }
            links = [None] * len(hands)
        jobs = [
            ([matchups[index] for index in hand], packed, settings, None if seed is None else seed + number, hand, messages, link)
            for number, (hand, link) in enumerate(zip(hands, links))
        ]
        try:
            if server is not None:
                pending = self._pool.map_async(_play_in_worker, jobs)
                server.serve(pending.ready, report_finished)
            elif messages is None:
                pending = None
                results = self._pool.map(_play_in_worker, jobs)
            else:
                pending = self._pool.map_async(_play_in_worker, jobs)
                while True:
                    try:
                        on_finished(*messages.get(timeout=0.1))
                    except queue.Empty:
                        if pending.ready():
                            break
            report_finished()
            if pending is not None:
                results = pending.get()
        finally:
            if server is not None:
                self.last_batching = (server.batches, server.positions)
                server.close()
        records: list[GameRecord | None] = [None] * len(matchups)
        for hand, played in zip(hands, results):
            for index, record in zip(hand, played):
                records[index] = record
        return records

    def close(self) -> None:
        self._pool.close()
        self._pool.join()
        if self._manager is not None:
            self._manager.shutdown()

    def __enter__(self) -> "RolloutPool":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


def replay(record: GameRecord) -> Game:
    """Play a recorded game again from its list of moves, and return the finished game."""

    game = Game(record.matchup.map)
    moves: dict[int, dict[int, SendAction]] = {}
    for tick, seat, source_id, target_id, units in record.actions:
        moves.setdefault(tick, {})[seat] = SendAction(source_id, target_id, units)
    while game.tick < record.ticks:
        game.step(moves.get(game.tick, {}))
    return game


def match_score(
    pool: "RolloutPool",
    first: str,
    second: str,
    maps: list[MapDefinition],
    models: dict[str, PolicyValueModel] | None = None,
    seed: int = 0,
    max_seconds: float = MATCH_SECONDS,
) -> float:
    """``first``'s score against ``second``: every map from both seats, wins 1 and draws 1/2."""

    matchups = []
    for definition in maps:
        matchups += [Matchup(definition, Seat(first), Seat(second)), Matchup(definition, Seat(second), Seat(first))]
    records = pool.play(matchups, models, seed=seed, max_seconds=max_seconds)
    points = 0.0
    for number, record in enumerate(records):
        seat = RED if number % 2 == 0 else BLUE
        points += 0.5 if record.winner is None else float(record.winner == seat)
    return points / len(records)


def count_decisions(records: list[GameRecord]) -> int:
    return sum(len(trajectory) for record in records for trajectory in record.trajectories)


class _Recorder:
    """Collects one seat's decisions while its game is being played."""

    def __init__(self, routes: np.ndarray):
        self.routes = routes  # the match's routes, as in MapLayout
        self.ticks: list[int] = []
        self.base_features: list[np.ndarray] = []
        self.global_features: list[np.ndarray] = []
        self.route_features: list[np.ndarray] = []
        self.allowed_bits: list[np.ndarray] = []
        self.route_limits: list[np.ndarray] = []
        self.choices: list[SendChoice | None] = []
        self.units: list[int] = []
        self.log_probabilities: list[float] = []
        self.values: list[float] = []
        self.captured: list[float] = []  # filled in by the game as each fleet lands

    def add(
        self,
        tick: int,
        state: EncodedState,
        legal: LegalActions,
        choice: SendChoice | None,
        units: int | None,
        log_p: float | None = None,
        value: float | None = None,
    ) -> None:
        self.ticks.append(tick)
        self.base_features.append(state.base_features)
        self.global_features.append(state.global_features)
        self.route_features.append(state.route_features[self.routes[:, 0], self.routes[:, 1]])
        self.allowed_bits.append(np.packbits(legal.allowed))
        self.route_limits.append(legal.limits[self.routes[:, 0], self.routes[:, 1]])
        self.choices.append(choice)
        self.units.append(WAIT if units is None else units)
        self.captured.append(np.nan)
        if log_p is not None:
            self.log_probabilities.append(log_p)
            self.values.append(value)

    def trajectory(self, seat: int, player: str, outcome: str, layout, by_model: bool, free: bool) -> Trajectory:
        bases = len(layout.base_ids)
        count = len(self.ticks)
        choices = np.array(
            [(WAIT, WAIT, WAIT) if choice is None else choice for choice in self.choices],
            dtype=np.int16,
        ).reshape(-1, 3)
        packed_bytes = -(-bases * bases * len(AMOUNT_OPTIONS) // 8)
        return Trajectory(
            seat=seat,
            player=player,
            outcome=outcome,
            base_ids=layout.base_ids,
            pair_features=layout.pair_features,
            # Explicit shapes, so a seat that never had a decision still gives well-formed arrays.
            base_features=np.array(self.base_features, dtype=np.float32).reshape(count, bases, len(BASE_FEATURES)),
            global_features=np.array(self.global_features, dtype=np.float32).reshape(count, len(GLOBAL_FEATURES)),
            routes=layout.routes,
            route_features=np.array(self.route_features, dtype=np.float32).reshape(
                count, len(layout.routes), len(ROUTE_FEATURES)
            ),
            allowed_bits=np.array(self.allowed_bits, dtype=np.uint8).reshape(count, packed_bytes),
            source=choices[:, 0],
            target=choices[:, 1],
            amount=choices[:, 2],
            tick=np.array(self.ticks, dtype=np.int32),
            log_probability=np.array(self.log_probabilities, dtype=np.float32) if by_model else None,
            value=np.array(self.values, dtype=np.float32) if by_model else None,
            captured=np.array(self.captured, dtype=np.float32),
            amount_options=len(AMOUNT_OPTIONS),
            units=np.array(self.units, dtype=np.int32),
            route_limits=np.array(self.route_limits, dtype=np.int32).reshape(count, len(layout.routes), len(LIMITS)),
            free_amounts=free,
        )


class _RunningGame:
    def __init__(self, index: int, matchup: Matchup, max_seconds: float):
        self.index = index
        self.matchup = matchup
        self.game = Game(matchup.map)
        self.max_ticks = round(max_seconds / self.game.seconds_per_tick)
        self.layout = build_layout(self.game.view_for(RED))
        self.seats = {RED: matchup.red, BLUE: matchup.blue}
        self.recorders = {seat: _Recorder(self.layout.routes) for seat, spec in self.seats.items() if spec.record}
        self.moves: dict[int, SendAction | None] = {}
        self.actions: list[tuple[int, int, str, str, int]] = []
        self.players: dict = {}  # scripted players by seat
        self.last_send: dict[int, int] = {}  # the tick of each seat's last accepted send
        # Recorded sends still in flight, by the tick at which they will have landed:
        # (seat, the decision's number in its recorder, target row).
        self.landings: dict[int, list[tuple[int, int, int]]] = {}
        seconds_per_tick = self.game.seconds_per_tick
        self.travel_ticks = {
            (source, target): ticks_until_arrival(seconds, seconds_per_tick)
            for source, routes in enumerate(self.layout.neighbours)
            for target, seconds in routes
        }

    def record(self, seat: int, tick: int, encoded, legal, choice, units, log_p=None, value=None) -> None:
        """Record a decision, and for a send, watch for its landing (see ``Trajectory.captured``)."""

        recorder = self.recorders[seat]
        recorder.add(tick, encoded, legal, choice, units, log_p, value)
        if choice is not None:
            landing = tick + self.travel_ticks[(choice.source, choice.target)]
            self.landings.setdefault(landing, []).append((seat, len(recorder.choices) - 1, choice.target))

    def may_send(self, seat: int) -> bool:
        """False while a seat with a send gap is still waiting after its last send."""

        last = self.last_send.get(seat)
        return last is None or self.game.tick - last >= self.seats[seat].min_send_gap

    def advance(self) -> bool:
        """Apply this tick's moves. Returns False once the game is over."""

        tick = self.game.tick
        for seat, action in self.moves.items():
            if action is not None:
                self.actions.append((tick, seat, action.source_id, action.target_id, action.units))
        rejected = self.game.step(self.moves)
        for seat, action in self.moves.items():
            if action is not None and seat not in rejected:
                self.last_send[seat] = tick
        for seat, decision, target in self.landings.pop(self.game.tick, ()):
            owner = self.game.bases[self.layout.base_ids[target]].owner
            self.recorders[seat].captured[decision] = float(owner == seat)
        return not self.game.is_finished() and self.game.tick < self.max_ticks

    def finish(self, model_names: set[str], free: set[str]) -> GameRecord:
        trajectories = [
            recorder.trajectory(
                seat,
                self.seats[seat].player,
                outcome_for(self.game, seat),
                self.layout,
                by_model=self.seats[seat].player in model_names,
                free=self.seats[seat].player in free,
            )
            for seat, recorder in self.recorders.items()
        ]
        return GameRecord(self.matchup, self.game.winner(), self.game.tick, self.actions, trajectories)


def _prepare_worker() -> None:
    # One thread per worker; the workers themselves already use every core.
    torch.set_num_threads(1)


def _play_in_worker(job) -> list[GameRecord]:
    matchups, packed, settings, seed, hand, messages, link = job
    models = {}
    for name, (model_name, config, weights) in packed.items():
        models[name] = build_model(model_name, **config)
        models[name].load_state_dict(weights)

    def report(index: int, record: GameRecord) -> None:
        messages.put((hand[index], [len(trajectory) for trajectory in record.trajectories], record.winner))

    report = report if messages is not None else None
    if link is None:
        return play_games(matchups, models, seed=seed, on_finished=report, **settings)
    inference = RemoteInference(link.connection, link.buffer_names, link.shape, link.players, link.free)
    try:
        return play_games(matchups, seed=seed, on_finished=report, inference=inference, **settings)
    finally:
        inference.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--red", default="untrained", help='"untrained", a checkpoint path, or an AI name')
    parser.add_argument("--blue", default="strategic_ai")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--first-seed", type=int, default=0, help="Map seed of the first game")
    parser.add_argument("--parallel-games", type=int, default=64, help="Games in progress at once, per process")
    parser.add_argument("--workers", type=int, default=1, help="Processes; 1 plays everything in this one")
    parser.add_argument("--device", default="cpu", help="Where models run when --workers is 1, e.g. mps")
    parser.add_argument("--decision-interval", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    from ml.models.checkpoint import init_model

    args = parse_args()
    models = {}
    for player in {args.red, args.blue}:
        if player == "untrained":
            models[player] = init_model("random", seed=0).model
        elif player.endswith(".pt"):
            models[player] = init_model(player).model

    matchups = [
        Matchup(generate_map(args.first_seed + index).definition, Seat(args.red, record=True), Seat(args.blue, record=True))
        for index in range(args.games)
    ]
    settings = dict(parallel_games=args.parallel_games, decision_interval=args.decision_interval, seed=0)

    started = time.perf_counter()
    if args.workers > 1:
        with RolloutPool(args.workers) as pool:
            ready = time.perf_counter()
            records = pool.play(matchups, models, **settings)
    else:
        ready = started
        records = play_games(matchups, models, device=args.device, **settings)
    seconds = time.perf_counter() - ready

    decisions = count_decisions(records)
    ticks = sum(record.ticks for record in records)
    red_wins = sum(record.winner == RED for record in records)
    blue_wins = sum(record.winner == BLUE for record in records)
    megabytes = sum(
        array.nbytes
        for record in records
        for trajectory in record.trajectories
        for array in (
            trajectory.base_features,
            trajectory.global_features,
            trajectory.route_features,
            trajectory.allowed_bits,
            trajectory.route_limits,
        )
    ) / 1e6

    print(f"{args.red} (red) vs {args.blue} (blue): {len(records)} games on generated maps")
    print(f"  red wins {red_wins}, blue wins {blue_wins}, draws {len(records) - red_wins - blue_wins}")
    print(f"  {ticks:,} game ticks, {decisions:,} recorded decisions, {megabytes:,.0f} MB of training data")
    print(f"  {seconds:.1f}s playing" + (f" (+{ready - started:.1f}s starting {args.workers} processes)" if args.workers > 1 else ""))
    print(f"  {decisions / seconds:,.0f} decisions/s, {len(records) / seconds:.1f} games/s")


if __name__ == "__main__":
    main()
