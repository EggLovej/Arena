"""Move scoring: try every move, keep the one whose result looks best.

The network's value output predicts who will win a position. This player
uses nothing else:

1. List every legal move: each send, and waiting.
2. For each, build the position right after it: the units leave the source
   and a fleet is on its way. The encoder already works out what fleets in
   flight will do when they land, so a send shows up in the position the
   moment it is made.
3. Ask the value output about all of these positions at once, in one batch.
4. Play the move whose position it likes best.

No look-ahead and no simulation, so it is about as fast as the plain network.
This is how TD-Gammon played backgammon: a learned judge of positions, and a
one-move look at every option. In theory each such step improves on the
player the judge learned from.

What it cannot see: the opponent's reply. It judges the position as if the
opponent keeps playing the way it did in training. And it will find the
positions the value output overrates, since it always picks the highest.

``candidates`` limits the moves it looks at to the policy output's favourite
sends (plus waiting); 0 looks at every legal move.

A player called ``value:PATH`` is understood by the rollouts and the
evaluator, for example ``value:ml/runs/latest.pt?candidates=8``.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from game_types import FleetView, GameView, SendAction

from ml.encoding.action_encoder import AMOUNT_OPTIONS, SendChoice, decode_action, legal_actions
from ml.encoding.forecast import build_timelines
from ml.encoding.state_encoder import encode_state, layout_for
from ml.models import PolicyValueModel, amounts, make_input


PREFIX = "value:"


@dataclass(frozen=True)
class ValueSettings:
    # Look only at the policy output's favourite sends, plus waiting; 0 looks at every legal move.
    candidates: int = 0
    # A separate judge (ml/models/judge.py) to judge the positions; empty: the network's own value output.
    judge: str = ""


class ValuePlayer:
    def __init__(self, model: PolicyValueModel, settings: ValueSettings = ValueSettings(), judge: PolicyValueModel | None = None):
        self.model = model.to("cpu").eval()
        # Proposes with the network's policy output, judges with the judge (by default the same network).
        self.judge = (judge or model).to("cpu").eval()
        self.settings = settings
        self.positions_judged = 0  # how many positions the value output has scored, for measuring cost

    @classmethod
    def from_init(cls, init: str | Path, settings: ValueSettings = ValueSettings()) -> "ValuePlayer":
        from ml.models.checkpoint import init_model
        from ml.play.players import game_path

        judge = init_model(game_path(settings.judge)).model if settings.judge else None
        return cls(init_model(init).model, settings, judge)

    def choose_action(self, state: GameView) -> SendAction | None:
        layout = layout_for(state)
        timelines = build_timelines(state)
        legal = legal_actions(state, layout, timelines)
        if not legal.allowed.any():
            return None

        moves = [None, *self._sends(state, layout, timelines, legal)]
        positions = [state, *(after_send(state, move) for move in moves[1:])]
        inputs = []
        for position in positions:
            position_timelines = build_timelines(position)
            inputs.append((encode_state(position, layout, position_timelines), legal_actions(position, layout, position_timelines)))
        with torch.no_grad():
            values = self.judge(make_input([item[0] for item in inputs], [item[1] for item in inputs])).value
        self.positions_judged += len(positions)
        # The first highest wins, so waiting is kept when a send is no better.
        return moves[int(torch.argmax(values))]

    __call__ = choose_action

    def _sends(self, state, layout, timelines, legal) -> list[SendAction]:
        if self.settings.candidates:
            return favourite_sends(self.model, state, layout, timelines, legal, self.settings.candidates)[0]
        choices = [SendChoice(*map(int, position)) for position in zip(*legal.allowed.nonzero())]
        return _distinct([decode_action(choice, legal) for choice in choices])


def favourite_sends(
    model, state, layout, timelines, legal, count: int, amounts_per_route: int = 1
) -> tuple[list[SendAction], list[float], float]:
    """The policy output's ``count`` most likely sends, each different, with their probabilities, and waiting's.

    For a network with free amounts see ``_free_favourites``; ``amounts_per_route`` is used only there.
    """

    with torch.no_grad():
        output = model(make_input([encode_state(state, layout, timelines)], [legal]))
    route_probability = torch.softmax(output.route_logits[0], dim=0)
    if output.amount_curves is not None:
        return _free_favourites(output, legal, route_probability, count, amounts_per_route)
    bases, options = len(legal.base_ids), len(AMOUNT_OPTIONS)
    joint = (route_probability[1:].view(bases, bases, 1) * torch.softmax(output.amount_logits[0], dim=-1)).flatten()
    joint[~torch.from_numpy(legal.allowed).flatten()] = 0.0
    sends, probabilities = [], []
    for index in joint.argsort(descending=True).tolist():
        if len(sends) == count or joint[index] <= 0:
            break
        choice = SendChoice(index // (bases * options), index // options % bases, index % options)
        send = decode_action(choice, legal)
        if send in sends:
            # Two amount options that come to the same units are the same move.
            probabilities[sends.index(send)] += float(joint[index])
            continue
        sends.append(send)
        probabilities.append(float(joint[index]))
    return sends, probabilities, float(route_probability[0])


def _free_favourites(output, legal, route_probability, count: int, per_route: int) -> tuple[list[SendAction], list[float], float]:
    """With free amounts: the most likely routes, each at its most likely amount.

    A free network spreads a route's probability over every amount from 1 to
    ``most``, so single amounts are not ranked against each other across routes
    (a route with a wide spread would lose to a narrow one however likely it
    is). Instead each route stands for itself, at the amount it likes best.
    With ``per_route`` above 1 a route also brings the amounts its other curves
    aim at (heaviest curve first, see ml/models/amounts.py), so fewer routes fit
    into ``count``. A route's probability is split over its amounts in
    proportion to how likely each is.
    """

    bases = len(legal.base_ids)
    sends, probabilities = [], []
    for column in route_probability[1:].argsort(descending=True).tolist():
        source, target = divmod(column, bases)
        probability = float(route_probability[1 + column])
        if len(sends) == count or probability <= 0:
            break
        most = int(legal.most[source, target])
        if most < 1:
            continue
        curves, most_units, needed = output.amount_curves[0, source, target], output.most[0, source, target], output.needed[0, source, target]
        grid = amounts.on_grid(curves, most_units, needed, most)  # the probability of 1, 2, ..., most units
        units = [int(grid.argmax()) + 1]
        if per_route > 1:
            log_weight, centre, _ = amounts.curves(curves, most_units, needed)
            for curve in log_weight.argsort(descending=True).tolist():
                aimed = min(max(round(float(centre[curve])), 1), most)
                if len(units) < per_route and aimed not in units:
                    units.append(aimed)
        units = units[: count - len(sends)]
        shares = grid[[amount - 1 for amount in units]]
        shares = shares / shares.sum() if shares.sum() > 0 else torch.full_like(shares, 1 / len(units))
        for amount, share in zip(units, shares.tolist()):
            sends.append(SendAction(source_id=legal.base_ids[source], target_id=legal.base_ids[target], units=amount))
            probabilities.append(probability * share)
    return sends, probabilities, float(route_probability[0])


def _distinct(sends: list[SendAction]) -> list[SendAction]:
    seen, kept = set(), []
    for send in sends:
        if send not in seen:
            seen.add(send)
            kept.append(send)
    return kept


def after_send(state: GameView, send: SendAction | None, owner: int | None = None) -> GameView:
    """The position right after this send: the units have left the source and are on their way.

    ``owner`` is who sends; by default the player the view belongs to. ``None`` as the send is waiting.
    """

    if send is None:
        return state
    owner = state.player_id if owner is None else owner
    travel = next(edge.travel_time for edge in state.edges if {edge.a, edge.b} == {send.source_id, send.target_id})
    bases = tuple(
        replace(base, units=base.units - send.units) if base.id == send.source_id else base for base in state.bases
    )
    fleet = FleetView(owner, send.units, send.source_id, send.target_id, travel, travel)
    return replace(state, bases=bases, fleets=(*state.fleets, fleet))
