"""Who runs the networks while games are played: each process itself, or one server on the GPU.

``play_games`` (ml/play/rollout.py) collects the positions where a network
has to decide and hands them to an inference object:

- ``LocalInference`` runs the networks right there, in the process that
  plays the games. A single process does this, and so does every worker of a
  ``RolloutPool`` by default, each on its own CPU core.
- With ``RolloutPool(inference_device="mps")`` the networks stay in the main
  process, on the GPU. The workers still simulate the games and encode the
  positions; they write them into shared memory and ask the server
  (``RemoteInference``). The server (``InferenceServer``) answers every
  request that is waiting with one forward pass, picks the moves on the GPU,
  and sends back only the choices. Each worker splits its games into two
  groups and simulates one while the server decides for the other.

Why: on one CPU core a network's cost is mostly fixed overhead per call, and
a larger network costs every worker. The GPU answers 192 positions of the
64x6 network in 7.5 ms, of the 128x8 one in 10 ms, and of a 256x8 one in
18 ms (measured 2026-10-03, M2 Pro).
"""

import copy
import math
import multiprocessing
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.shared_memory import SharedMemory
from collections.abc import Callable

import numpy as np
import torch

from ml.encoding.action_encoder import AMOUNT_OPTIONS, LegalActions, SendChoice
from ml.encoding.state_encoder import BASE_FEATURES, GLOBAL_FEATURES, PAIR_FEATURES, ROUTE_FEATURES, EncodedState
from ml.models import ModelInput, ModelOutput, PolicyValueModel, amounts, make_input
from ml.models.policy import _amount_logits_for, _free_amounts_for, choose, log_probability, to_choices


WAITED = -1  # in the server's answers, in place of a source, target, and amount
# Below this many positions, a GPU server answers on the CPU instead (measured 2026-10-03: on
# one core, 16 positions of the 64x6 network take 2.9 ms; any GPU call takes about 4 ms).
SMALL_BATCH = 16


@dataclass
class Request:
    """The positions of one group of games, at one tick, where a network has to decide."""

    states: list[EncodedState]
    legal: list[LegalActions]
    players: list[str]  # which network decides each position


@dataclass
class Answers:
    """One choice per position of a request, with how likely it was and the position's value.

    For a network with free amounts (listed in the inference object's ``free``), a choice's
    amount is the units; see ``decode_action``.
    """

    choices: list[SendChoice | None]
    log_probabilities: list[float]
    values: list[float]


class LocalInference:
    """The networks run in this process, one batch per network and request."""

    groups = 1  # nothing to overlap: the answer is ready as soon as it is asked for

    def __init__(
        self,
        models: dict[str, PolicyValueModel],
        device: torch.device | str = "cpu",
        greedy: bool = False,
        generator: torch.Generator | None = None,
        temperatures: dict[str, float] | None = None,
    ):
        self.device = device
        self.models = {name: model.to(device).eval() for name, model in models.items()}
        self.names = set(self.models)
        self.free = {name for name, model in self.models.items() if model.free_amounts}
        self.greedy = greedy
        self.generator = generator
        self.temperatures = check_temperatures(temperatures)

    def submit(self, group: int, request: Request) -> Answers:
        count = len(request.states)
        answers = Answers([None] * count, [0.0] * count, [0.0] * count)
        # One batch per network, in the order the networks first appear.
        for name in dict.fromkeys(request.players):
            rows = [row for row, player in enumerate(request.players) if player == name]
            inputs = make_input([request.states[row] for row in rows], [request.legal[row] for row in rows], self.device)
            with torch.no_grad():
                output = self.models[name](inputs)
            # Choosing happens on the CPU so one seeded generator works on every device.
            output = output.to("cpu")
            route, amount = choose(output, self.greedy, self.generator, self.temperatures.get(name, 1.0))
            choices = to_choices(route, amount, bases=inputs.base_mask.shape[1])
            for row, choice, log_p, value in zip(
                rows, choices, log_probability(output, route, amount).tolist(), output.value.tolist()
            ):
                answers.choices[row] = choice
                answers.log_probabilities[row] = log_p
                answers.values[row] = value
        return answers

    def collect(self, ticket: Answers) -> Answers:
        return ticket


# ---- Shared memory between a worker and the server ---------------------------------------

@dataclass(frozen=True)
class BufferShape:
    """The size of one request buffer: room for ``rows`` positions of up to ``bases`` bases."""

    rows: int
    bases: int

    def arrays(self) -> dict[str, tuple[tuple[int, ...], type]]:
        rows, bases = self.rows, self.bases
        return {
            "base_features": ((rows, bases, len(BASE_FEATURES)), np.float32),
            "pair_features": ((rows, bases, bases, len(PAIR_FEATURES)), np.float32),
            "route_features": ((rows, bases, bases, len(ROUTE_FEATURES)), np.float32),
            "global_features": ((rows, len(GLOBAL_FEATURES)), np.float32),
            "base_mask": ((rows, bases), np.bool_),
            "allowed": ((rows, bases, bases, len(AMOUNT_OPTIONS)), np.bool_),
            "most": ((rows, bases, bases), np.float32),
            "needed": ((rows, bases, bases), np.float32),
            "player": ((rows,), np.int16),
        }

    def size(self) -> int:
        return sum(_padded(math.prod(shape) * np.dtype(kind).itemsize) for shape, kind in self.arrays().values())


# How many of each array's axes (after the first, the rows) run over bases.
BASE_AXES = {
    "base_features": 1, "pair_features": 2, "route_features": 2, "global_features": 0,
    "base_mask": 1, "allowed": 2, "most": 2, "needed": 2, "player": 0,
}  # fmt: skip


def _padded(size: int) -> int:
    return -(-size // 8) * 8


class RequestBuffer:
    """Named numpy arrays over one block of shared memory."""

    def __init__(self, memory: SharedMemory, shape: BufferShape, owner: bool):
        self.memory = memory
        self.owner = owner
        self.arrays = {}
        offset = 0
        for name, (dimensions, kind) in shape.arrays().items():
            self.arrays[name] = np.ndarray(dimensions, dtype=kind, buffer=memory.buf, offset=offset)
            offset += _padded(math.prod(dimensions) * np.dtype(kind).itemsize)

    @classmethod
    def create(cls, shape: BufferShape) -> "RequestBuffer":
        return cls(SharedMemory(create=True, size=shape.size()), shape, owner=True)

    @classmethod
    def attach(cls, name: str, shape: BufferShape) -> "RequestBuffer":
        return cls(SharedMemory(name=name), shape, owner=False)

    def write(self, request: Request, player_numbers: dict[str, int]) -> int:
        """Put a request's positions into rows 0, 1, ...; returns the largest map's base count."""

        arrays = self.arrays
        largest = 0
        for row, (state, legal, player) in enumerate(zip(request.states, request.legal, request.players)):
            count = len(state.base_ids)
            largest = max(largest, count)
            # The whole row is cleared first, so padding is always zero (and never allowed).
            for array in arrays.values():
                array[row] = 0
            arrays["base_features"][row, :count] = state.base_features
            arrays["pair_features"][row, :count, :count] = state.pair_features
            arrays["route_features"][row, :count, :count] = state.route_features
            arrays["global_features"][row] = state.global_features
            arrays["base_mask"][row, :count] = True
            arrays["allowed"][row, :count, :count] = legal.allowed
            arrays["most"][row, :count, :count] = legal.most
            arrays["needed"][row, :count, :count] = legal.needed
            arrays["player"][row] = player_numbers[player]
        return largest

    def close(self) -> None:
        self.arrays = {}
        self.memory.close()
        if self.owner:
            self.memory.unlink()


# ---- The worker's side ----------------------------------------------------------------

class RemoteInference:
    """In a worker: ask the server in the main process, through shared memory and a pipe."""

    groups = 2  # while the server decides for one group of games, the worker simulates the other

    def __init__(
        self,
        connection: Connection,
        buffer_names: list[str],
        shape: BufferShape,
        players: tuple[str, ...],
        free: tuple[str, ...] = (),
    ):
        self.connection = connection
        self.buffers = [RequestBuffer.attach(name, shape) for name in buffer_names]
        self.names = set(players)
        self.free = set(free)
        self.numbers = {player: number for number, player in enumerate(players)}

    def submit(self, group: int, request: Request) -> tuple[int, int]:
        count = len(request.states)
        if count:
            largest = self.buffers[group].write(request, self.numbers)
            self.connection.send((group, count, largest))
        return group, count

    def collect(self, ticket: tuple[int, int]) -> Answers:
        group, count = ticket
        if not count:
            return Answers([], [], [])
        # The server answers each worker's requests in the order they were sent.
        answered_group, source, target, amount, log_probabilities, values = self.connection.recv()
        assert answered_group == group and len(source) == count
        choices = [
            None if s == WAITED else SendChoice(int(s), int(t), int(a)) for s, t, a in zip(source, target, amount)
        ]
        return Answers(choices, log_probabilities.tolist(), values.tolist())

    def close(self) -> None:
        for buffer in self.buffers:
            buffer.close()
        self.connection.close()


# ---- The server's side ----------------------------------------------------------------

@dataclass(frozen=True)
class WorkerLink:
    """What one worker job needs to reach the server."""

    connection: Connection
    buffer_names: list[str]
    shape: BufferShape
    players: tuple[str, ...]
    free: tuple[str, ...]  # the players with free amounts


class InferenceServer:
    """In the main process: answer the workers' requests in batches, on ``device``.

    Use as ``links = server.open(...)``, hand one link to each worker job, then
    ``server.serve(...)`` until the jobs are done, and ``server.close()``.
    """

    def __init__(
        self,
        models: dict[str, PolicyValueModel],
        device: torch.device | str,
        greedy: bool = False,
        seed: int | None = None,
        temperatures: dict[str, float] | None = None,
    ):
        self.device = torch.device(device)
        self.players = tuple(models)
        # How much each player's odds are sharpened: 1 / its temperature (see ``policy.choose``).
        temperatures = check_temperatures(temperatures)
        self.sharpen = [1.0 / temperatures.get(player, 1.0) for player in self.players]
        self.models = []
        self._training = []  # (model, whether it was in training mode), restored by ``close``
        for model in models.values():
            on_device = next(model.parameters()).device.type == self.device.type
            # A model elsewhere gets a copy here, so the caller's model stays where it is.
            served = model if on_device else copy.deepcopy(model).to(self.device)
            self._training.append((model, model.training))
            self.models.append(served.eval())
        # Copies for small batches, when the server runs on a GPU.
        self.cpu_models = None if self.device.type == "cpu" else [copy.deepcopy(model).cpu() for model in self.models]
        self.greedy = greedy
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(0 if seed is None else seed)
        self.cpu_generator = torch.Generator()
        self.cpu_generator.manual_seed(1 if seed is None else seed + 1)
        self.shape: BufferShape | None = None  # set by ``open``
        self.connections: list[Connection] = []
        self.buffers: list[list[RequestBuffer]] = []  # per worker job, one buffer per group
        self.batches = self.positions = 0  # for measuring how well requests are batched

    def open(self, jobs: int, shape: BufferShape, groups: int = RemoteInference.groups) -> list[WorkerLink]:
        self.shape = shape
        links = []
        for _ in range(jobs):
            mine, theirs = multiprocessing.get_context("spawn").Pipe()
            buffers = [RequestBuffer.create(shape) for _ in range(groups)]
            self.connections.append(mine)
            self.buffers.append(buffers)
            free = tuple(player for player, model in zip(self.players, self.models) if model.free_amounts)
            links.append(WorkerLink(theirs, [buffer.memory.name for buffer in buffers], shape, self.players, free))
        return links

    def serve(self, done: Callable[[], bool], meanwhile: Callable[[], None] = lambda: None) -> None:
        """Answer requests until ``done()`` says the workers have finished. ``meanwhile`` runs between batches."""

        open_connections = list(self.connections)
        while True:
            requests = []
            for connection in wait(open_connections, timeout=0.01):
                # A worker can have two requests waiting, one per group.
                while True:
                    try:
                        if not connection.poll():
                            break
                        group, count, largest = connection.recv()
                    except (EOFError, OSError):
                        open_connections.remove(connection)
                        break
                    requests.append((connection, group, count, largest))
            if requests:
                self._answer(requests)
            meanwhile()
            if not requests and done():
                return

    def _answer(self, requests: list[tuple[Connection, int, int, int]]) -> None:
        total = sum(request[2] for request in requests)
        # A handful of positions (the last few long games of a round) is quicker on the CPU:
        # every GPU call costs about 4 ms, however small.
        on_cpu = self.cpu_models is not None and total < SMALL_BATCH
        if on_cpu:
            bases = max(request[3] for request in requests)
        else:
            # On the GPU, every new batch shape costs time to prepare, so batches come in a few
            # shapes only: maps padded to the round's largest, row counts rounded up.
            bases = self.shape.bases
        parts = {name: [] for name in BASE_AXES}
        for connection, group, count, _ in requests:
            arrays = self.buffers[self.connections.index(connection)][group].arrays
            for name, axes in BASE_AXES.items():
                parts[name].append(arrays[name][(slice(count),) + (slice(bases),) * axes])
        stacked = {name: np.concatenate(arrays) for name, arrays in parts.items()}
        players = stacked.pop("player")

        results = np.zeros((total, 4), dtype=np.float32)  # route, amount, log-probability, value
        for number in np.unique(players):
            rows = slice(None) if len(self.players) == 1 else np.flatnonzero(players == number)
            chosen = {name: array[rows] for name, array in stacked.items()}
            count = len(chosen["base_mask"])
            if not on_cpu and (padded := _batch_size(count)) > count:
                # The extra rows repeat the first position; their answers are thrown away.
                chosen = {name: np.concatenate((array, np.repeat(array[:1], padded - count, axis=0))) for name, array in chosen.items()}
            device = torch.device("cpu") if on_cpu else self.device
            inputs = ModelInput(**{name: torch.from_numpy(array) for name, array in chosen.items()}).to(device)
            model = (self.cpu_models if on_cpu else self.models)[number]
            with torch.no_grad():
                output = model(inputs)
                route, amount = self._choose(output, self.cpu_generator if on_cpu else self.generator, self.sharpen[number])
                answer = torch.stack(
                    (route.float(), amount.float(), log_probability(output, route, amount), output.value), dim=1
                )
            results[rows] = answer[:count].cpu().numpy()

        route, amount = results[:, 0].astype(np.int64), results[:, 1].astype(np.int64)
        waited = route == 0
        source = np.where(waited, WAITED, (route - 1) // bases).astype(np.int16)
        target = np.where(waited, WAITED, (route - 1) % bases).astype(np.int16)
        amount = np.where(waited, WAITED, amount).astype(np.int16)
        start = 0
        for connection, group, count, _ in requests:
            end = start + count
            connection.send((group, source[start:end], target[start:end], amount[start:end], results[start:end, 2], results[start:end, 3]))
            start = end
        self.batches += 1
        self.positions += total

    def _choose(self, output: ModelOutput, generator: torch.Generator, sharpen: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """One choice per position, drawn where the output is (Gumbel-max: the same distribution as sampling).

        ``sharpen`` is 1 / the temperature: the odds are raised to that power, as in ``policy.choose``.
        """

        route_logits = output.route_logits * sharpen
        if not self.greedy:
            route_logits = route_logits + _gumbel(route_logits.shape, generator)
        route = route_logits.argmax(dim=1)
        if output.amount_curves is not None:
            curves, most, needed = _free_amounts_for(output, route)
            if self.greedy:
                amount = amounts.most_likely(curves, most, needed)
            elif sharpen != 1.0:
                amount = amounts.sample_sharpened(curves, most, needed, sharpen, generator)
            else:
                amount = amounts.sample(curves, most, needed, generator)
            return route, torch.where(route == 0, 0, amount)
        amount_logits = _amount_logits_for(output, route) * sharpen
        if not self.greedy:
            amount_logits = amount_logits + _gumbel(amount_logits.shape, generator)
        amount = amount_logits.argmax(dim=1)
        return route, torch.where(route == 0, 0, amount)

    def close(self) -> None:
        for connection in self.connections:
            connection.close()
        for buffers in self.buffers:
            for buffer in buffers:
                buffer.close()
        self.connections, self.buffers = [], []
        for model, training in self._training:
            model.train(training)


def _batch_size(count: int) -> int:
    """``count`` rounded up to one of a few sizes: 16, 24, 32, 48, 64, 96, ... (two per doubling)."""

    step = 2 ** max(2, (count - 1).bit_length() - 2)
    return max(16, -(-count // step) * step)


def check_temperatures(temperatures: dict[str, float] | None) -> dict[str, float]:
    """Temperatures by player name (see ``play_games``); every one must be above 0."""

    temperatures = dict(temperatures or {})
    for player, temperature in temperatures.items():
        if temperature <= 0:
            raise ValueError(f"temperature must be above 0, not {temperature} (for {player})")
    return temperatures


def _gumbel(shape: torch.Size, generator: torch.Generator) -> torch.Tensor:
    uniform = torch.rand(shape, generator=generator, device=generator.device).clamp(min=1e-10, max=1 - 1e-7)
    return -torch.log(-torch.log(uniform))
