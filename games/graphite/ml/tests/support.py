"""Helpers that only the tests need."""

from dataclasses import fields

import numpy as np
import torch

from ml.models import ModelInput
from ml.models.interface import ModelOutput
from ml.play.rollout import WAIT, Trajectory


def trajectory_input(
    trajectory: Trajectory,
    decisions: slice | np.ndarray = slice(None),
    device: torch.device | str = "cpu",
) -> ModelInput:
    """Recorded decisions as a batch the model can be run on again."""

    base_features = torch.from_numpy(trajectory.base_features[decisions])
    count, bases = base_features.shape[:2]
    limits = None if trajectory.route_limits is None else trajectory.limit_table(decisions).astype(np.float32)
    return ModelInput(
        base_features=base_features,
        pair_features=torch.tensor(trajectory.pair_features).expand(count, -1, -1, -1),
        route_features=torch.from_numpy(trajectory.route_table(decisions)),
        global_features=torch.from_numpy(trajectory.global_features[decisions]),
        base_mask=torch.ones(count, bases, dtype=torch.bool),
        allowed=torch.from_numpy(trajectory.allowed(decisions)),
        most=None if limits is None else torch.from_numpy(limits[..., 0].copy()),
        needed=None if limits is None else torch.from_numpy(limits[..., 1].copy()),
    ).to(device)


def trajectory_choices(
    trajectory: Trajectory,
    decisions: slice | np.ndarray = slice(None),
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """The recorded choices in the form ``ml.models.policy`` works with (units, for free amounts)."""

    source = trajectory.source[decisions].astype(np.int64)
    target = trajectory.target[decisions].astype(np.int64)
    amount = trajectory.amount[decisions].astype(np.int64)
    waited = source == WAIT
    route = np.where(waited, 0, 1 + source * len(trajectory.base_ids) + target)
    return (
        torch.from_numpy(route).to(device),
        torch.from_numpy(np.where(waited, 0, amount)).to(device),
    )


def repeated(output: ModelOutput, row: int, count: int) -> ModelOutput:
    """One position's output ``count`` times over, to draw from it many times at once."""

    parts = {field.name: getattr(output, field.name) for field in fields(output)}
    return ModelOutput(**{name: None if part is None else part[row : row + 1].expand(count, *part.shape[1:]) for name, part in parts.items()})
