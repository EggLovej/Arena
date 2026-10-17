"""Two-move look-ahead: our move and the opponent's reply, judged together.

Move scoring (``value_player.py``) looks at our move alone, as if the
opponent were not about to move as well. In this game both players move at
the same moment, so the honest question is: which of my moves is best,
given what the opponent might do at the same time?

1. Take our few most likely moves (from the policy output, plus waiting),
   and the opponent's few most likely moves, from the same network looking
   at the game from the opponent's side: it assumes the opponent plays as
   well as it does itself. A network with free amounts offers its most
   likely routes, each at its most likely amount (``amounts`` tries more
   amounts per route).
2. For every pair, build the position with both fleets on their way, and
   let the value output judge all of them in one batch: a small table of
   our moves against theirs.
3. Choose our move from the table by one of three rules (``rule``):

   - ``worst``: the move whose worst case is best (assumes the opponent
     knows our move, which overrates it).
   - ``expected``: the best average, weighting the opponent's moves by how
     likely the network thinks the opponent is to play each.
   - ``nash``: the game-theory answer for moves made at the same time. Find
     the mix of our moves that no mix of theirs can exploit, and play the
     move that mix uses most.

Like move scoring it needs no simulation, because the encoder works out what
fleets in flight will do. A table of 9 by 9 positions costs about 10 to 30 ms.

A player called ``lookahead:PATH`` is understood by the rollouts and the
evaluator, for example ``lookahead:ml/runs/latest.pt?mine=8&theirs=8&rule=nash``;
``&judge=PATH`` fills the table with a separate judge (``ml/models/judge.py``).
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from game_types import GameView, SendAction

from ml.encoding.action_encoder import legal_actions
from ml.encoding.forecast import build_timelines
from ml.encoding.state_encoder import encode_state, layout_for
from ml.models import PolicyValueModel, make_input
from ml.play.value_player import after_send, favourite_sends


PREFIX = "lookahead:"
RULES = ("worst", "expected", "nash")


@dataclass(frozen=True)
class LookaheadSettings:
    mine: int = 8  # our most likely sends looked at, besides waiting
    theirs: int = 8  # the opponent's most likely sends looked at, besides waiting
    rule: str = "nash"  # how to choose from the table: worst, expected, or nash
    # For a network with free amounts: the amounts tried per route, its most likely first, then
    # those its other curves aim at; the sends still number ``mine`` and ``theirs`` in all.
    amounts: int = 1
    # A separate judge (ml/models/judge.py) to fill the table; empty: the network's own value output.
    judge: str = ""


class LookaheadPlayer:
    def __init__(
        self,
        model: PolicyValueModel,
        settings: LookaheadSettings = LookaheadSettings(),
        judge: PolicyValueModel | None = None,
    ):
        if settings.rule not in RULES:
            raise ValueError(f"rule must be one of {', '.join(RULES)}, not '{settings.rule}'")
        self.model = model.to("cpu").eval()
        # The network's policy output proposes both sides' moves; the judge fills the table.
        self.judge = (judge or model).to("cpu").eval()
        self.settings = settings
        self.positions_judged = 0

    @classmethod
    def from_init(cls, init: str | Path, settings: LookaheadSettings = LookaheadSettings()) -> "LookaheadPlayer":
        from ml.models.checkpoint import init_model

        judge = init_model(settings.judge).model if settings.judge else None
        return cls(init_model(init).model, settings, judge)

    def choose_action(self, state: GameView) -> SendAction | None:
        table = self.table(state)
        if table is None:
            return None
        return table.our_moves[choose_row(table.values, self.settings.rule, table.their_weights)]

    __call__ = choose_action

    def table(self, state: GameView) -> "Table | None":
        """The whole table for this position; None when waiting is the only legal move."""

        layout = layout_for(state)
        timelines = build_timelines(state)
        legal = legal_actions(state, layout, timelines)
        if not legal.allowed.any():
            return None
        ours, probabilities, wait = favourite_sends(
            self.model, state, layout, timelines, legal, self.settings.mine, self.settings.amounts
        )
        our_moves = [None, *ours]

        opponents = {base.owner for base in state.bases if base.owner not in (0, state.player_id)}
        opponents |= {fleet.owner for fleet in state.fleets if fleet.owner != state.player_id}
        if len(opponents) != 1:
            # No opponent left, or more than one: judge our moves alone, as move scoring does.
            their_moves, their_weights, enemy = [None], np.ones(1), None
        else:
            (enemy,) = opponents
            their_moves, their_weights = self._their_moves(replace(state, player_id=enemy), layout)

        values = self._judge(state, layout, our_moves, their_moves, enemy)
        return Table(our_moves, np.array([wait, *probabilities]), their_moves, their_weights, values, legal)

    def _their_moves(self, their_view: GameView, layout) -> tuple[list[SendAction | None], np.ndarray]:
        """The opponent's likely moves as our network sees them, and how likely each is (normalised)."""

        timelines = build_timelines(their_view)
        legal = legal_actions(their_view, layout, timelines)
        if not legal.allowed.any():
            return [None], np.ones(1)
        sends, probabilities, wait = favourite_sends(
            self.model, their_view, layout, timelines, legal, self.settings.theirs, self.settings.amounts
        )
        weights = np.array([wait, *probabilities])
        return [None, *sends], weights / weights.sum()

    def _judge(self, state, layout, our_moves, their_moves, enemy) -> np.ndarray:
        """The value output for every pair of moves: rows are ours, columns are theirs."""

        inputs = []
        for ours in our_moves:
            after_ours = after_send(state, ours)
            for theirs in their_moves:
                position = after_send(after_ours, theirs, owner=enemy)
                timelines = build_timelines(position)
                inputs.append((encode_state(position, layout, timelines), legal_actions(position, layout, timelines)))
        with torch.no_grad():
            values = self.judge(make_input([item[0] for item in inputs], [item[1] for item in inputs])).value
        self.positions_judged += len(inputs)
        return values.numpy().reshape(len(our_moves), len(their_moves))


@dataclass
class Table:
    """Everything the look-ahead worked out for one position."""

    our_moves: list[SendAction | None]  # waiting first, then our most likely sends
    our_probabilities: np.ndarray  # how likely the network thought each of our moves was
    their_moves: list[SendAction | None]
    their_weights: np.ndarray  # how likely the network thinks each of their moves is (adds up to 1)
    values: np.ndarray  # [ours, theirs]: the judge's value of each pair of moves, -1 to +1
    legal: object  # the position's LegalActions

    def scores(self, rule: str) -> np.ndarray:
        """How good each of our moves is under the rule: what it earns against the opponent's mix."""

        if rule == "worst":
            return self.values.min(axis=1)
        if rule == "expected":
            return self.values @ self.their_weights
        _, theirs = solve_matrix_game(self.values)
        return self.values @ theirs


def choose_row(table: np.ndarray, rule: str, their_weights: np.ndarray) -> int:
    """Our move (a row of the table) by the given rule. The first best wins, so waiting wins ties."""

    if rule == "worst":
        return int(np.argmax(table.min(axis=1)))
    if rule == "expected":
        return int(np.argmax(table @ their_weights))
    if rule == "nash":
        ours, _ = solve_matrix_game(table)
        return int(np.argmax(ours))
    raise ValueError(f"unknown rule '{rule}'")


def solve_matrix_game(table: np.ndarray, rounds: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Mixed strategies for a game where we pick a row to make the value high and they a column to make it low.

    Both sides repeatedly shift weight towards the moves they regret not
    having played (regret matching); the average of what they played
    approaches a Nash equilibrium of the zero-sum game.
    """

    rows, columns = table.shape
    regret_ours, regret_theirs = np.zeros(rows), np.zeros(columns)
    sum_ours, sum_theirs = np.zeros(rows), np.zeros(columns)
    for _ in range(rounds):
        ours = _from_regrets(regret_ours)
        theirs = _from_regrets(regret_theirs)
        sum_ours += ours
        sum_theirs += theirs
        row_values = table @ theirs  # what each of our moves would have earned
        column_values = ours @ table  # what each of theirs would have conceded
        expected = ours @ row_values
        # "Plus": regrets never go below zero, which makes the averages settle much faster.
        regret_ours = np.maximum(regret_ours + row_values - expected, 0.0)
        regret_theirs = np.maximum(regret_theirs + expected - column_values, 0.0)
    return sum_ours / sum_ours.sum(), sum_theirs / sum_theirs.sum()


def _from_regrets(regrets: np.ndarray) -> np.ndarray:
    total = regrets.sum()
    return regrets / total if total > 0 else np.full(len(regrets), 1.0 / len(regrets))
