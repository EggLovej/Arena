"""A small transformer that treats every base as a token.

How a position flows through it:

1. Each base's features become a list of ``width`` numbers, its "description".
   The global features are added to every base, so each one knows the score.
2. Several attention layers let every base look at every other base and
   update its description. How strongly base A listens to base B is learned,
   and nudged by what we know about the pair: is there a route, how long is
   it, how far apart are they, and what would a capture along it cost now.
3. To score a route, the source's description, the target's description, and
   the pair's features are combined. One small layer (or more, see
   ``route_layers``) turns that into a route score and a score for each
   amount option, or, with ``amounts="free"``, a few curves over every whole
   number of units (ml/models/amounts.py). Only routes that can be used right now are scored; that is
   a handful out of hundreds of base pairs. With ``capture_head``, the same
   layer also says, per amount, whether the fleet would leave the target ours.
4. The descriptions are averaged into one summary of the position, which
   gives the score for waiting and the value (who is winning).

Nothing depends on the order of the bases or on how many there are, so one
network plays every map.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

from ml.encoding.action_encoder import AMOUNT_OPTIONS, JUST_ENOUGH
from ml.models import amounts as free_amounts
from ml.encoding.state_encoder import (
    COLUMNS_BEFORE_V2,
    GLOBAL_FEATURES,
    PAIR_FEATURES,
    feature_columns,
)
from ml.models.interface import ILLEGAL, ModelInput, PolicyValueModel, Scores, padded_route_list


@dataclass(frozen=True)
class GraphTransformerConfig:
    width: int = 64  # how many numbers describe each base inside the network
    layers: int = 3  # rounds of bases looking at each other
    heads: int = 4  # separate "questions" a base can ask per round; must divide width
    # Hidden layers that judge each usable route before it is scored. Every
    # send decision goes through them, so this is where "where to send, and
    # how much" is worked out.
    route_layers: int = 1
    # Input and output sizes. They follow the encoders and are stored with a
    # saved model, so an old model still loads after the encoders change: the
    # model reads the first this many of each feature list and ignores any
    # added later. Set lower to train without the newest features. Base and route
    # columns stop before the 2026-10-08 additions (``COLUMNS_BEFORE_V2``), which
    # are meant to be read by name.
    base_features: int = COLUMNS_BEFORE_V2["base_features"]
    pair_features: int = len(PAIR_FEATURES)
    route_features: int = COLUMNS_BEFORE_V2["route_features"]
    global_features: int = len(GLOBAL_FEATURES)
    amount_options: int = len(AMOUNT_OPTIONS)
    # An extra output per route and amount: will the target be ours right after the fleet
    # lands? Learned from what happened in the games (``Scores.capture``). Off for every
    # model saved before it existed.
    capture_head: bool = False
    # How the amount is chosen: "options" (the AMOUNT_OPTIONS) or "free" (any whole number of
    # units, from ``amount_curves`` curves per route; added 2026-10-08).
    amounts: str = "options"
    amount_curves: int = free_amounts.CURVES
    # Which columns to read: "" reads the first ``base_features`` (and so on) of each list, as
    # every network before 2026-10-09; the name of a set in FEATURE_SETS (ml/encoding/state_encoder.py,
    # e.g. "v2") reads exactly its columns, by name, and the four counts above follow the set.
    feature_set: str = ""

    def __post_init__(self):
        if self.feature_set:
            for table, columns in feature_columns(self.feature_set).items():
                object.__setattr__(self, table, len(columns))


class GraphTransformer(PolicyValueModel):
    def __init__(self, config: GraphTransformerConfig = GraphTransformerConfig()):
        super().__init__()
        self.config = config
        width = config.width
        if config.route_layers < 1:
            raise ValueError("route_layers must be at least 1")
        if config.amounts not in ("options", "free"):
            raise ValueError(f"amounts must be 'options' or 'free', not '{config.amounts}'")
        if config.amounts == "free" and config.capture_head:
            raise ValueError("the capture output is per amount option, so it needs amounts='options'")
        # Per route, after its score: one score per amount option, or each curve's parameters.
        self.amount_outputs = (
            config.amount_options if config.amounts == "options" else config.amount_curves * free_amounts.PARAMETERS
        )
        # A named feature set reads its columns by position in each table.
        if config.feature_set:
            for table, columns in feature_columns(config.feature_set).items():
                self.register_buffer(f"_{table}_columns", torch.tensor(columns, dtype=torch.long), persistent=False)
        # The fixed route table, what a capture along each route costs now, and
        # one more live flag: can this route send "just enough" right now?
        pair_inputs = config.pair_features + config.route_features + 1

        self.embed_bases = nn.Linear(config.base_features, width)
        self.embed_globals = nn.Linear(config.global_features, width)
        self.pair_bias = nn.Sequential(
            nn.Linear(pair_inputs, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, config.layers * config.heads),
        )
        self.blocks = nn.ModuleList(AttentionBlock(width, config.heads) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(width)

        self.as_source = nn.Linear(width, width)
        self.as_target = nn.Linear(width, width)
        self.as_pair = nn.Linear(pair_inputs, width)
        # With route_layers 1 this is empty and passes its input through unchanged.
        self.route_thinking = nn.Sequential(
            *(module for _ in range(config.route_layers - 1) for module in (nn.Linear(width, width), nn.ReLU()))
        )
        self.route_out = nn.Linear(width, 1 + self.amount_outputs)
        self.capture_out = nn.Linear(width, config.amount_options) if config.capture_head else None
        wait_final = nn.Linear(width, 1)
        value_final = nn.Linear(width, 1)
        self.wait_out = nn.Sequential(nn.Linear(width, width), nn.ReLU(), wait_final)
        self.value_out = nn.Sequential(nn.Linear(width, width), nn.ReLU(), value_final)

        # Start undecided: near-equal scores for every choice and a value near zero.
        for layer in (self.route_out, wait_final, value_final):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        if config.amounts == "free":
            # Except the curves: they start on "just enough", "half" and "everything".
            with torch.no_grad():
                self.route_out.bias[1:] = free_amounts.starting_biases(config.amount_curves).flatten()

    def scores(self, inputs: ModelInput) -> Scores:
        config = self.config
        batch, bases = inputs.base_mask.shape

        can_send_just_enough = inputs.allowed[..., JUST_ENOUGH : JUST_ENOUGH + 1].float()
        pairs = torch.cat(
            (
                self._read(inputs.pair_features, "pair_features"),
                self._read(inputs.route_features, "route_features"),
                can_send_just_enough,
            ),
            dim=-1,
        )

        # One attention nudge per layer and head, and no listening to padding.
        bias = self.pair_bias(pairs).view(batch, bases, bases, config.layers, config.heads)
        bias = bias.permute(3, 0, 4, 1, 2)  # [layers, batch, heads, listener, speaker]
        bias = bias.masked_fill(~inputs.base_mask[None, :, None, None, :], ILLEGAL)

        described = self.embed_bases(self._read(inputs.base_features, "base_features"))
        described = described + self.embed_globals(self._read(inputs.global_features, "global_features"))[:, None, :]
        for block, block_bias in zip(self.blocks, bias):
            described = block(described, block_bias)
        described = self.final_norm(described)

        # Score only the usable routes, then place the results in the full
        # table. Every other entry stays 0 and is removed as illegal anyway.
        position, source, target = padded_route_list(inputs.allowed.any(dim=-1), batch)
        # One extra, never-read position receives the padding (see padded_route_list).
        described = torch.cat((described, described[:1]), dim=0)
        pairs = torch.cat((pairs, pairs[:1]), dim=0)
        usable = torch.relu(
            self.as_source(described)[position, source]
            + self.as_target(described)[position, target]
            + self.as_pair(pairs[position, source, target])
        )
        judged = self.route_thinking(usable)
        route = described.new_zeros(batch + 1, bases, bases, 1 + self.amount_outputs)
        route[position, source, target] = self.route_out(judged)
        route, described = route[:batch], described[:batch]
        capture = None
        if self.capture_out is not None:
            capture = described.new_zeros(batch + 1, bases, bases, config.amount_options)
            capture[position, source, target] = self.capture_out(judged)
            capture = capture[:batch]

        # Average over real bases only, so padding never changes the answer.
        real = inputs.base_mask[:, :, None].float()
        summary = (described * real).sum(dim=1) / real.sum(dim=1)

        free = config.amounts == "free"
        return Scores(
            wait=self.wait_out(summary)[:, 0],
            route=route[..., 0],
            amount=None if free else route[..., 1:],
            value=self.value_out(summary)[:, 0],
            capture=capture,
            amount_curves=route[..., 1:].unflatten(-1, (config.amount_curves, free_amounts.PARAMETERS)) if free else None,
        )


    def _read(self, values: Tensor, table: str) -> Tensor:
        """The columns of one input table that this network reads."""

        if self.config.feature_set:
            return values.index_select(-1, getattr(self, f"_{table}_columns"))
        return values[..., : getattr(self.config, table)]


class AttentionBlock(nn.Module):
    """One round of every base looking at every other base, then thinking it over."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("width must be a multiple of heads")
        self.heads = heads
        self.norm_before_attention = nn.LayerNorm(width)
        self.query_key_value = nn.Linear(width, 3 * width)
        self.after_attention = nn.Linear(width, width)
        self.norm_before_thinking = nn.LayerNorm(width)
        self.thinking = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))

    def forward(self, described: Tensor, bias: Tensor) -> Tensor:
        batch, bases, width = described.shape
        # Each base forms a query (what am I looking for), a key (what do I
        # offer), and a value (what I say when listened to), once per head.
        parts = self.query_key_value(self.norm_before_attention(described))
        parts = parts.view(batch, bases, 3, self.heads, width // self.heads)
        query, key, value = parts.permute(2, 0, 3, 1, 4)

        heard = scaled_dot_product_attention(query, key, value, attn_mask=bias)
        heard = heard.transpose(1, 2).reshape(batch, bases, width)

        described = described + self.after_attention(heard)
        return described + self.thinking(self.norm_before_thinking(described))
