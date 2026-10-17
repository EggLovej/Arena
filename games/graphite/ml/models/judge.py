"""The judge: a network that only says who will win from a position.

It reads a position exactly as the playing network does (same encoder, same
attention over bases) but has no moves to choose. Instead it gives three
probabilities that add up to 1: the player to move wins, draws, or loses.

Trained on its own (see ``ml/judge/``), on many more positions than a
self-play round has, and without having to share its weights with a
policy, it is meant to tell apart positions that differ by a single move:
what move scoring and the look-ahead ask of a judge.

It fits the common model interface, so it saves, loads, and batches like
any model. Its ``value`` is the expected result on the value output's scale,
P(win) − P(loss), from −1 to +1; its move scores are all equal, since it does
not choose moves.
"""

from dataclasses import dataclass, replace
from math import log1p

import torch
from torch import Tensor, nn

from ml.encoding.action_encoder import AMOUNT_OPTIONS, JUST_ENOUGH
from ml.encoding.state_encoder import BASE_FEATURES, COLUMNS_BEFORE_V2, GLOBAL_FEATURES, PAIR_FEATURES, ROUTE_FEATURES, UNITS_REFERENCE
from ml.models.graph_transformer import AttentionBlock
from ml.models.interface import ILLEGAL, ModelInput, ModelOutput, PolicyValueModel, Scores


OUTCOMES = ("win", "draw", "loss")
# The encoder gives unit counts on a log scale; these are the columns that hold them.
UNIT_COLUMNS = tuple(BASE_FEATURES.index(name) for name in ("units", "incoming_mine", "incoming_enemy", "future_units"))
ROUTE_UNIT_COLUMNS = (ROUTE_FEATURES.index("capture_cost"),)


@dataclass(frozen=True)
class GraphJudgeConfig:
    width: int = 64
    layers: int = 6
    heads: int = 4
    # As in the playing network: the model reads the first this many of each feature list.
    base_features: int = COLUMNS_BEFORE_V2["base_features"]
    pair_features: int = len(PAIR_FEATURES)
    route_features: int = COLUMNS_BEFORE_V2["route_features"]
    global_features: int = len(GLOBAL_FEATURES)
    amount_options: int = len(AMOUNT_OPTIONS)
    # "log": unit counts as the encoder gives them, log(1 + units) scaled. "linear": the judge turns
    # them back into plain counts (divided by 250) before reading them, so each count is seen once,
    # in the form fights use: a fight is a subtraction.
    unit_scale: str = "log"


class GraphJudge(PolicyValueModel):
    def __init__(self, config: GraphJudgeConfig = GraphJudgeConfig()):
        super().__init__()
        self.config = config
        width = config.width
        pair_inputs = config.pair_features + config.route_features + 1

        # The same parts, under the same names, as graph_transformer's, so a
        # playing network's understanding of positions can be copied in as a start.
        self.embed_bases = nn.Linear(config.base_features, width)
        self.embed_globals = nn.Linear(config.global_features, width)
        self.pair_bias = nn.Sequential(
            nn.Linear(pair_inputs, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, config.layers * config.heads),
        )
        self.blocks = nn.ModuleList(AttentionBlock(width, config.heads) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(width)
        # The average of the bases and the largest of each number: "how it is overall" and
        # "the biggest threat or opportunity", which an average alone blurs.
        self.outcome_out = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, len(OUTCOMES)))

    def outcome_logits(self, inputs: ModelInput) -> Tensor:
        """Scores for win, draw, and loss: [batch, 3]. Softmax turns them into probabilities."""

        config = self.config
        batch, bases = inputs.base_mask.shape
        base_features = inputs.base_features[..., : config.base_features]
        route_features = inputs.route_features[..., : config.route_features]
        if config.unit_scale == "linear":
            base_features = _linear_units(base_features, UNIT_COLUMNS)
            route_features = _linear_units(route_features, ROUTE_UNIT_COLUMNS)
        can_send_just_enough = inputs.allowed[..., JUST_ENOUGH : JUST_ENOUGH + 1].float()
        pairs = torch.cat((inputs.pair_features[..., : config.pair_features], route_features, can_send_just_enough), dim=-1)
        bias = self.pair_bias(pairs).view(batch, bases, bases, config.layers, config.heads)
        bias = bias.permute(3, 0, 4, 1, 2).masked_fill(~inputs.base_mask[None, :, None, None, :], ILLEGAL)

        described = self.embed_bases(base_features)
        described = described + self.embed_globals(inputs.global_features[..., : config.global_features])[:, None, :]
        for block, block_bias in zip(self.blocks, bias):
            described = block(described, block_bias)
        described = self.final_norm(described)

        real = inputs.base_mask[:, :, None]
        average = (described * real).sum(dim=1) / real.sum(dim=1)
        largest = described.masked_fill(~real, ILLEGAL).max(dim=1).values
        return self.outcome_out(torch.cat((average, largest), dim=1))

    def outcome_probabilities(self, inputs: ModelInput) -> Tensor:
        return torch.softmax(self.outcome_logits(inputs), dim=1)

    def scores(self, inputs: ModelInput) -> Scores:
        # No preferences between moves: the judge only judges positions.
        batch, bases = inputs.base_mask.shape
        zeros = inputs.base_features.new_zeros
        return Scores(
            wait=zeros(batch),
            route=zeros(batch, bases, bases),
            amount=zeros(batch, bases, bases, self.config.amount_options),
            value=zeros(batch),
        )

    def forward(self, inputs: ModelInput) -> ModelOutput:
        output = super().forward(inputs)
        win, _, loss = self.outcome_probabilities(inputs).unbind(dim=1)
        return replace(output, value=win - loss)


def _linear_units(features: Tensor, columns: tuple[int, ...]) -> Tensor:
    """The same features with the given log-scaled unit columns as plain counts / UNITS_REFERENCE."""

    columns = [column for column in columns if column < features.shape[-1]]
    if not columns:
        return features
    features = features.clone()
    log_scale = log1p(UNITS_REFERENCE)
    features[..., columns] = torch.expm1(features[..., columns] * log_scale) / UNITS_REFERENCE
    return features
