"""Recorded decisions, arranged for training.

Rollouts return trajectories: one seat's decisions in one game. Training
wants something else: shuffled batches of single decisions drawn from many
games. A ``DecisionSet`` holds decisions that way.

Maps differ in size, so decisions are kept in groups by number of bases and
every batch comes from one group. That avoids padding, and each group is
drawn from as often as its share of the decisions.

Each decision can carry extra numbers, its "targets", under any name a
trainer likes: the game's outcome for imitation, or advantages for self-play.

For a network with free amounts (``DecisionSet(free_amounts=True)``), a
send's amount is the units it sent, and each batch carries the most and the
needed of every route, which that network reads.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

from ml.encoding.action_encoder import AMOUNT_OPTIONS, option_units
from ml.models import ModelInput
from ml.play.rollout import WAIT, Trajectory, spread_routes


@dataclass
class Batch:
    inputs: ModelInput
    route: torch.Tensor  # [batch], 0 for wait, as ml.models.policy uses it
    amount: torch.Tensor  # [batch], the amount option, or with free amounts the units
    targets: dict[str, torch.Tensor]  # each [batch]
    # With ``DecisionSet(option_units=True)``: what every amount option would have sent on
    # every route, 0 where not on offer. [batch, bases, bases, amount options]
    option_units: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.route)


@dataclass
class _Group:
    """All decisions made on maps with one particular number of bases."""

    bases: int
    # Filled trajectory by trajectory, then joined into single arrays by ``_join``.
    base_features: list = field(default_factory=list)
    global_features: list = field(default_factory=list)
    allowed_bits: list = field(default_factory=list)
    route: list = field(default_factory=list)
    amount: list = field(default_factory=list)
    trajectory: list = field(default_factory=list)  # which pair_features and routes entry each decision uses
    pair_features: list = field(default_factory=list)  # one entry per trajectory
    # Maps of the same size can have different numbers of routes. Both lists
    # are padded to the longest when joined; a padding route points from base
    # 0 to itself, which is never a real route, and carries zeros.
    routes: list = field(default_factory=list)  # one entry per trajectory
    route_features: list = field(default_factory=list)
    route_limits: list = field(default_factory=list)  # only where the set needs them
    targets: dict[str, list] = field(default_factory=dict)
    joined: dict[str, np.ndarray] | None = None

    def join(self) -> dict[str, np.ndarray]:
        if self.joined is None:
            longest = max(len(routes) for routes in self.routes)
            self.joined = {
                "base_features": np.concatenate(self.base_features),
                "global_features": np.concatenate(self.global_features),
                "allowed_bits": np.concatenate(self.allowed_bits),
                "route": np.concatenate(self.route),
                "amount": np.concatenate(self.amount),
                "trajectory": np.concatenate(self.trajectory),
                "pair_features": np.stack(self.pair_features),
                "routes": np.stack([_pad_routes(routes, longest) for routes in self.routes]),
                "route_features": np.concatenate([_pad_routes(values, longest, axis=1) for values in self.route_features]),
                **(
                    {"route_limits": np.concatenate([_pad_routes(values, longest, axis=1) for values in self.route_limits])}
                    if self.route_limits
                    else {}
                ),
                **{f"target:{name}": np.concatenate(parts) for name, parts in self.targets.items()},
            }
        return self.joined


class DecisionSet:
    def __init__(self, free_amounts: bool = False, option_units: bool = False):
        """``free_amounts``: for a network with free amounts. ``option_units``: each batch also
        says what every amount option would have sent (for teaching such a network from one
        with amount options)."""

        self.free_amounts = free_amounts
        self.option_units = option_units
        self._groups: dict[int, _Group] = {}
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def add(
        self,
        trajectory: Trajectory,
        targets: dict[str, np.ndarray] | None = None,
        max_decisions: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Add one trajectory's decisions and return how many were added.

        ``targets`` holds one value per decision under each name. With
        ``max_decisions``, a longer trajectory contributes a random sample of
        that size, so long games do not crowd out short ones.
        """

        count = len(trajectory)
        if count == 0:
            return 0
        limits = self.free_amounts or self.option_units
        if trajectory.free_amounts and not self.free_amounts:
            raise ValueError("A network with amount options cannot learn from choices made with free amounts")
        keep = np.arange(count)
        if max_decisions is not None and count > max_decisions:
            rng = rng or np.random.default_rng()
            keep = np.sort(rng.choice(count, size=max_decisions, replace=False))

        bases = len(trajectory.base_ids)
        group = self._groups.setdefault(bases, _Group(bases))
        group.joined = None

        source = trajectory.source[keep].astype(np.int64)
        waited = source == WAIT
        group.base_features.append(trajectory.base_features[keep])
        group.global_features.append(trajectory.global_features[keep])
        group.allowed_bits.append(trajectory.allowed_bits[keep])
        group.route.append(np.where(waited, 0, 1 + source * bases + trajectory.target[keep]))
        amount = trajectory.units if self.free_amounts else trajectory.amount
        group.amount.append(np.where(waited, 0, amount[keep]).astype(np.int64))
        group.trajectory.append(np.full(len(keep), len(group.pair_features), dtype=np.int64))
        group.pair_features.append(trajectory.pair_features)
        group.routes.append(trajectory.routes)
        group.route_features.append(trajectory.route_features[keep])
        if limits:
            group.route_limits.append(trajectory.route_limits[keep])

        targets = targets or {}
        if group.targets and set(targets) != set(group.targets):
            raise ValueError("Every trajectory in a DecisionSet must carry the same targets")
        for name, values in targets.items():
            group.targets.setdefault(name, []).append(np.asarray(values, dtype=np.float32)[keep])

        self._count += len(keep)
        return len(keep)

    def batches(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
        device: torch.device | str = "cpu",
        equal_size: bool = False,
    ) -> Iterator[Batch]:
        """One pass over every decision, in shuffled batches. Pass no ``rng`` to keep the stored order.

        With ``equal_size``, a group's last, shorter batch is filled up with
        decisions drawn again at random from the same group, so every batch
        has exactly ``batch_size`` decisions. On the Apple GPU every new batch
        shape costs memory that is never given back, and a training run that
        sees new shapes every round slowly fills the machine.
        """

        plan = []
        for bases, group in self._groups.items():
            order = np.arange(len(group.join()["route"]))
            if rng is not None:
                rng.shuffle(order)
            chunks = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
            if equal_size and len(chunks[-1]) < batch_size:
                picker = rng or np.random.default_rng(0)
                extra = picker.choice(order, size=batch_size - len(chunks[-1]), replace=True)
                chunks[-1] = np.concatenate((chunks[-1], extra))
            plan += [(bases, chunk) for chunk in chunks]
        if rng is not None:
            rng.shuffle(plan)

        for bases, chosen in plan:
            yield self._batch(self._groups[bases], np.sort(chosen), device)

    def _batch(self, group: _Group, chosen: np.ndarray, device: torch.device | str) -> Batch:
        data = group.join()
        bases = group.bases
        bits = np.unpackbits(data["allowed_bits"][chosen], axis=1, count=bases * bases * len(AMOUNT_OPTIONS))
        routes = data["routes"][data["trajectory"][chosen]]
        limits = spread_routes(data["route_limits"][chosen], routes, bases) if "route_limits" in data else None
        inputs = ModelInput(
            base_features=torch.from_numpy(data["base_features"][chosen]),
            pair_features=torch.from_numpy(data["pair_features"][data["trajectory"][chosen]]),
            route_features=torch.from_numpy(spread_routes(data["route_features"][chosen], routes, bases)),
            global_features=torch.from_numpy(data["global_features"][chosen]),
            base_mask=torch.ones(len(chosen), bases, dtype=torch.bool),
            allowed=torch.from_numpy(bits.reshape(len(chosen), bases, bases, -1).astype(bool)),
            most=None if limits is None else torch.from_numpy(limits[..., 0].astype(np.float32)),
            needed=None if limits is None else torch.from_numpy(limits[..., 1].astype(np.float32)),
        )
        return Batch(
            inputs=inputs.to(device),
            route=torch.from_numpy(data["route"][chosen]).to(device),
            amount=torch.from_numpy(data["amount"][chosen]).to(device),
            targets={
                name.removeprefix("target:"): torch.from_numpy(values[chosen]).to(device)
                for name, values in data.items()
                if name.startswith("target:")
            },
            option_units=torch.from_numpy(option_units(limits)).to(device) if self.option_units else None,
        )


def _pad_routes(array: np.ndarray, length: int, axis: int = 0) -> np.ndarray:
    """Pad a list of routes (or per-route values) with zeros to ``length`` along ``axis``."""

    padding = [(0, 0)] * array.ndim
    padding[axis] = (0, length - array.shape[axis])
    return np.pad(array, padding)
