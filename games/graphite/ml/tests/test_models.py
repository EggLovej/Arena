import unittest
from dataclasses import replace

import torch

from game import Game
from game_types import BLUE, RED
from loading import load_ai

from ml.encoding.action_encoder import AMOUNT_OPTIONS, LegalActions, decode_action, legal_actions
from ml.encoding.state_encoder import BASE_FEATURES, ROUTE_FEATURES, EncodedState, encode_state, feature_columns
from ml.map_generator import generate_map
from ml.models import MODELS, build_model, make_input
from ml.models.amounts import CURVES, PARAMETERS, on_grid
from ml.models.interface import ILLEGAL, ModelInput, ModelOutput, PolicyValueModel, Scores, padded_route_list
from ml.models.policy import choose, entropy, entropy_parts, log_probability, to_choices
from ml.tests.support import repeated


def positions(seeds=range(6), every=40, ticks=200):
    """Encoded mid-game positions from maps of different sizes."""

    strategic = load_ai("strategic_ai")
    aggressive = load_ai("aggressive_ai")
    states, legal = [], []
    for seed in seeds:
        game = Game(generate_map(seed).definition)
        for tick in range(ticks):
            if game.is_finished():
                break
            game.step({RED: strategic(game.view_for(RED)), BLUE: aggressive(game.view_for(BLUE))})
            if tick % every == every - 1:
                state = game.view_for(RED if tick % 2 else BLUE)
                states.append(encode_state(state))
                legal.append(legal_actions(state))
    return states, legal


def check_route_temperature(test: unittest.TestCase, output: ModelOutput) -> None:
    """Routes drawn at a temperature follow the odds raised to 1 / temperature; 1 changes nothing."""

    count = 4000
    # Clearly different odds for the usable routes of the first position, so sharpening shows.
    logits = output.route_logits[0].clone()
    usable = logits > ILLEGAL
    logits[usable] = torch.linspace(0.0, 2.0, int(usable.sum()))
    many = replace(repeated(output, 0, count), route_logits=logits.expand(count, -1))

    plain = choose(many, generator=torch.Generator().manual_seed(1))
    at_one = choose(many, generator=torch.Generator().manual_seed(1), temperature=1.0)
    test.assertTrue(torch.equal(plain[0], at_one[0]) and torch.equal(plain[1], at_one[1]))

    route, _ = choose(many, generator=torch.Generator().manual_seed(1), temperature=0.5)
    expected = torch.softmax(logits * 2, dim=0)
    counts = torch.bincount(route, minlength=len(expected)).float() / count
    spread = (expected * (1 - expected) / count).sqrt()
    test.assertTrue(((counts - expected).abs() <= 5 * spread + 1e-3).all())
    favourite = int(logits.argmax())
    test.assertGreater(counts[favourite], (plain[0] == favourite).float().mean() + 0.05)
    with test.assertRaises(ValueError):
        choose(output, temperature=0)


def reordered(state: EncodedState, legal: LegalActions, order: list[int]):
    """The same position with its bases listed in a different order."""

    return (
        EncodedState(
            base_ids=tuple(state.base_ids[i] for i in order),
            base_features=state.base_features[order],
            pair_features=state.pair_features[order][:, order],
            route_features=state.route_features[order][:, order],
            global_features=state.global_features,
        ),
        LegalActions(
            base_ids=tuple(legal.base_ids[i] for i in order),
            allowed=legal.allowed[order][:, order],
            units=legal.units[order][:, order],
            limits=legal.limits[order][:, order],
        ),
    )


class EverythingEqualModel(PolicyValueModel):
    """The smallest possible architecture: no preferences at all."""

    def __init__(self):
        super().__init__()
        self.level = torch.nn.Parameter(torch.zeros(()))

    def scores(self, inputs: ModelInput) -> Scores:
        batch, bases = inputs.base_mask.shape
        flat = self.level.expand(batch)
        return Scores(
            wait=flat,
            route=self.level.expand(batch, bases, bases),
            amount=self.level.expand(batch, bases, bases, len(AMOUNT_OPTIONS)),
            value=flat,
        )


class FreeAmountModelTests(unittest.TestCase):
    """A network that picks any whole number of units keeps the same rules as an option network."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.states, cls.legal = positions(seeds=range(3))
        cls.inputs = make_input(cls.states, cls.legal)
        cls.model = build_model(amounts="free").eval()
        with torch.no_grad():
            cls.output = cls.model(cls.inputs)

    def test_it_gives_curves_instead_of_option_scores(self):
        self.assertTrue(self.model.free_amounts)
        self.assertFalse(build_model().free_amounts)
        self.assertIsNone(self.output.amount_logits)
        batch, bases = self.inputs.base_mask.shape
        self.assertEqual(self.output.amount_curves.shape, (batch, bases, bases, CURVES, PARAMETERS))
        with self.assertRaises(ValueError):
            build_model(amounts="free", capture_head=True)
        with self.assertRaises(ValueError):
            build_model(amounts="any")

    def test_it_may_use_exactly_the_routes_an_option_network_may_use(self):
        usable = (self.inputs.most > 0).flatten(1)
        self.assertTrue(torch.equal(usable, self.inputs.allowed.any(dim=-1).flatten(1)))
        route_probability = torch.softmax(self.output.route_logits, dim=1)[:, 1:]
        self.assertEqual(route_probability[~usable].max(), 0.0)

    def test_every_sampled_choice_is_a_legal_move(self):
        bases = self.inputs.base_mask.shape[1]
        for _ in range(20):
            route, amount = choose(self.output)
            for choice, legal in zip(to_choices(route, amount, bases), self.legal):
                decode_action(choice, legal, free=True)  # raises if the choice is not legal

    def test_probabilities_of_all_choices_sum_to_one_and_entropy_agrees(self):
        # Every choice of every position: wait, or a usable route and any amount from 1 to its most.
        output, batch, bases = self.output, *self.inputs.base_mask.shape
        rows, routes, units = [torch.arange(batch)], [torch.zeros(batch, dtype=torch.long)], [torch.zeros(batch, dtype=torch.long)]
        for row, source, target in (self.inputs.most > 0).nonzero().tolist():
            most = int(self.inputs.most[row, source, target])
            rows.append(torch.full((most,), row))
            routes.append(torch.full((most,), 1 + source * bases + target))
            units.append(torch.arange(1, most + 1))
        rows, routes, units = torch.cat(rows), torch.cat(routes), torch.cat(units)
        each = ModelOutput(
            route_logits=output.route_logits[rows],
            amount_logits=None,
            value=output.value[rows],
            amount_curves=output.amount_curves[rows],
            most=output.most[rows],
            needed=output.needed[rows],
        )
        chance = log_probability(each, routes, units).exp()
        total = torch.zeros(batch).index_add(0, rows, chance)
        self.assertTrue(torch.allclose(total, torch.ones(batch), atol=1e-4))
        direct = torch.zeros(batch).index_add(0, rows, -chance * torch.log(chance.clamp(min=1e-30)))
        self.assertTrue(torch.allclose(entropy(output), direct, atol=1e-3))

    def test_entropy_parts_are_the_route_choice_and_the_rest(self):
        # The whole is checked against every choice above; so the amount part, being the rest,
        # is the amounts' entropy given the route, weighted by how likely the route is.
        route_part, amount_part = entropy_parts(self.output)
        route_probability = torch.softmax(self.output.route_logits, dim=1)
        direct = -(route_probability * torch.log(route_probability.clamp(min=1e-30))).sum(dim=1)
        self.assertTrue(torch.allclose(route_part, direct, atol=1e-5))
        self.assertTrue(torch.allclose(route_part + amount_part, entropy(self.output)))
        self.assertTrue((amount_part > 0).all())

    def test_a_temperature_sharpens_the_odds_of_the_route_and_the_amount(self):
        check_route_temperature(self, self.output)
        # One route only, with room for several amounts: its amounts follow the sharpened odds.
        count, bases = 4000, self.inputs.base_mask.shape[1]
        source, target = max(((s, t) for s, t in (self.inputs.most[0] > 0).nonzero().tolist()), key=lambda pair: self.inputs.most[0][pair])
        logits = torch.full_like(self.output.route_logits[0], ILLEGAL)
        logits[1 + source * bases + target] = 0.0
        many = replace(repeated(self.output, 0, count), route_logits=logits.expand(count, -1))
        _, amount = choose(many, generator=torch.Generator().manual_seed(2), temperature=0.5)
        most = int(self.output.most[0, source, target])
        self.assertGreater(most, 2)
        odds = on_grid(self.output.amount_curves[0, source, target], self.output.most[0, source, target], self.output.needed[0, source, target], most)
        expected = odds**2 / (odds**2).sum()
        counts = torch.bincount(amount, minlength=most + 1)[1:].float() / count
        spread = (expected * (1 - expected) / count).sqrt()
        self.assertTrue(((counts - expected).abs() <= 5 * spread + 1e-3).all())

    def test_greedy_takes_the_likeliest_route_then_its_likeliest_amount(self):
        route, amount = choose(self.output, greedy=True)
        self.assertTrue(torch.equal(route, self.output.route_logits.argmax(dim=1)))
        bases = self.inputs.base_mask.shape[1]
        for row in (route > 0).nonzero()[:, 0].tolist():
            source, target = divmod(route[row].item() - 1, bases)
            most = self.output.most[row, source, target]
            table = on_grid(self.output.amount_curves[row, source, target], most, self.output.needed[row, source, target], int(most))
            self.assertEqual(amount[row].item(), table.argmax().item() + 1)

    def test_base_order_does_not_matter(self):
        generator = torch.Generator().manual_seed(1)
        for state, legal in zip(self.states[:6], self.legal[:6]):
            count = len(state.base_ids)
            order = torch.randperm(count, generator=generator).tolist()
            with torch.no_grad():
                original = self.model(make_input([state], [legal]))
                shuffled = self.model(make_input(*[[x] for x in reordered(state, legal, order)]))
            curves = original.amount_curves[0][order][:, order]
            self.assertTrue(torch.allclose(shuffled.amount_curves[0], curves, atol=1e-4))
            self.assertTrue(torch.equal(shuffled.most[0], original.most[0][order][:, order]))
            self.assertTrue(torch.equal(shuffled.needed[0], original.needed[0][order][:, order]))

    def test_the_curves_learn_from_the_probability_of_a_choice(self):
        model = build_model(amounts="free")
        output = model(self.inputs)
        route, amount = choose(output)
        (-log_probability(output, route, amount).mean() - entropy(output).mean()).backward()
        self.assertGreater(model.route_out.weight.grad[1:].abs().sum().item(), 0.0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.states, cls.legal = positions()
        cls.inputs = make_input(cls.states, cls.legal)
        cls.model = build_model().eval()
        with torch.no_grad():
            cls.output = cls.model(cls.inputs)

    def test_registry_builds_every_model(self):
        for name in MODELS:
            self.assertIsInstance(build_model(name), PolicyValueModel)
        with self.assertRaises(ValueError):
            build_model("no_such_model")

    def test_settings_change_the_model(self):
        small = sum(p.numel() for p in build_model(width=32, layers=2).parameters())
        default = sum(p.numel() for p in build_model().parameters())
        self.assertLess(small, default)
        self.assertGreater(default, 50_000)
        self.assertLess(default, 300_000)

    def test_extra_route_layers_add_weights_and_keep_the_rules(self):
        deeper = build_model(route_layers=3).eval()
        added = sum(p.numel() for p in deeper.parameters()) - sum(p.numel() for p in self.model.parameters())
        width = deeper.config.width
        self.assertEqual(added, 2 * (width * width + width))
        with torch.no_grad():
            output = deeper(self.inputs)
        # Still undecided at the start, and still nothing illegal.
        self.assertLess(output.value.abs().max(), 0.2)
        route_allowed = self.inputs.allowed.any(dim=-1).flatten(1)
        self.assertEqual(torch.softmax(output.route_logits, dim=1)[:, 1:][~route_allowed].max(), 0.0)
        with self.assertRaises(ValueError):
            build_model(route_layers=0)

    def test_a_model_reads_only_the_features_it_was_built_for(self):
        torch.manual_seed(0)
        old = build_model(route_features=0, global_features=4).eval()
        new = build_model().eval()
        changed = replace(
            self.inputs,
            route_features=torch.rand_like(self.inputs.route_features),
            global_features=torch.cat(
                (self.inputs.global_features[:, :4], torch.rand_like(self.inputs.global_features[:, 4:])), dim=1
            ),
        )
        with torch.no_grad():
            self.assertTrue(torch.equal(old(self.inputs).route_logits, old(changed).route_logits))
            self.assertFalse(torch.equal(new(self.inputs).route_logits, new(changed).route_logits))

    def test_output_shapes(self):
        batch, bases = self.inputs.base_mask.shape
        self.assertEqual(self.output.route_logits.shape, (batch, 1 + bases * bases))
        self.assertEqual(self.output.amount_logits.shape, (batch, bases, bases, len(AMOUNT_OPTIONS)))
        self.assertEqual(self.output.value.shape, (batch,))

    def test_untrained_model_is_undecided(self):
        self.assertLess(self.output.value.abs().max(), 0.2)
        route_probability = torch.softmax(self.output.route_logits, dim=1)
        routes = self.inputs.allowed.any(dim=-1).flatten(1).sum(dim=1)
        # Waiting is roughly one choice among all the legal routes.
        expected = 1 / (1 + routes)
        self.assertTrue(torch.allclose(route_probability[:, 0], expected, rtol=0.3))

    def test_illegal_choices_have_zero_probability(self):
        route_probability = torch.softmax(self.output.route_logits, dim=1)[:, 1:]
        route_allowed = self.inputs.allowed.any(dim=-1).flatten(1)
        self.assertEqual(route_probability[~route_allowed].max(), 0.0)
        amount_probability = torch.softmax(self.output.amount_logits, dim=-1)
        on_legal_routes = amount_probability[self.inputs.allowed.any(dim=-1)]
        allowed_on_legal_routes = self.inputs.allowed[self.inputs.allowed.any(dim=-1)]
        self.assertEqual(on_legal_routes[~allowed_on_legal_routes].max(), 0.0)

    def test_every_sampled_choice_is_a_legal_move(self):
        bases = self.inputs.base_mask.shape[1]
        for _ in range(20):
            route, amount = choose(self.output)
            for choice, legal in zip(to_choices(route, amount, bases), self.legal):
                decode_action(choice, legal)  # raises if the choice is not legal

    def test_greedy_picks_the_most_likely_choice(self):
        route, amount = choose(self.output, greedy=True)
        self.assertTrue(torch.equal(route, self.output.route_logits.argmax(dim=1)))
        again, _ = choose(self.output, greedy=True)
        self.assertTrue(torch.equal(route, again))
        # Greedy is the most likely route, then the most likely amount for it.
        # That is not always the most likely pair: waiting has no amount to
        # choose, so it can beat a slightly likelier route split nine ways.
        route_log_p = torch.log_softmax(self.output.route_logits, dim=1)
        best = route_log_p.gather(1, route[:, None])
        for _ in range(5):
            sampled, _ = choose(self.output)
            self.assertTrue((route_log_p.gather(1, sampled[:, None]) <= best + 1e-5).all())
        rows = torch.arange(len(route))
        sends = route > 0
        source, target = (route[sends] - 1) // self.output.amount_logits.shape[1], (route[sends] - 1) % self.output.amount_logits.shape[1]
        amount_logits = self.output.amount_logits[rows[sends], source, target]
        self.assertTrue(torch.equal(amount[sends], amount_logits.argmax(dim=1)))

    def test_a_temperature_sharpens_the_odds_of_the_route_and_the_amount(self):
        check_route_temperature(self, self.output)
        count, bases = 4000, self.inputs.base_mask.shape[1]
        allowed = self.inputs.allowed[0]
        source, target = max(((s, t) for s, t in allowed.any(dim=-1).nonzero().tolist()), key=lambda pair: int(allowed[pair].sum()))
        route_logits = torch.full_like(self.output.route_logits[0], ILLEGAL)
        route_logits[1 + source * bases + target] = 0.0
        amount_logits = self.output.amount_logits[0].clone()
        legal = amount_logits[source, target] > ILLEGAL
        amount_logits[source, target, legal] = torch.linspace(0.0, 2.0, int(legal.sum()))
        many = replace(
            repeated(self.output, 0, count),
            route_logits=route_logits.expand(count, -1),
            amount_logits=amount_logits.expand(count, *amount_logits.shape),
        )
        _, amount = choose(many, generator=torch.Generator().manual_seed(2), temperature=0.5)
        expected = torch.softmax(amount_logits[source, target] * 2, dim=0)
        counts = torch.bincount(amount, minlength=len(expected)).float() / count
        spread = (expected * (1 - expected) / count).sqrt()
        self.assertTrue(((counts - expected).abs() <= 5 * spread + 1e-3).all())

    def test_probabilities_of_all_choices_sum_to_one(self):
        batch, bases = self.inputs.base_mask.shape
        options = len(AMOUNT_OPTIONS)
        total = log_probability(
            self.output,
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
        ).exp()
        joint = [total]
        for column in range(1, 1 + bases * bases):
            for option in range(options):
                route = torch.full((batch,), column)
                amount = torch.full((batch,), option)
                joint.append(log_probability(self.output, route, amount).exp())
        joint = torch.stack(joint, dim=1)
        self.assertTrue(torch.allclose(joint.sum(dim=1), torch.ones(batch), atol=1e-4))

        # Entropy must agree with its definition over that full list of choices.
        direct = -(joint * torch.log(joint.clamp(min=1e-30))).sum(dim=1)
        self.assertTrue(torch.allclose(entropy(self.output), direct, atol=1e-3))

    def test_entropy_parts_are_the_route_choice_and_the_rest(self):
        route_part, amount_part = entropy_parts(self.output)
        route_probability = torch.softmax(self.output.route_logits, dim=1)
        direct = -(route_probability * torch.log(route_probability.clamp(min=1e-30))).sum(dim=1)
        self.assertTrue(torch.allclose(route_part, direct, atol=1e-5))
        self.assertTrue(torch.allclose(route_part + amount_part, entropy(self.output)))
        # A position where only waiting is likely has (almost) no amount part.
        certain = ModelOutput(
            route_logits=torch.cat((torch.full_like(self.output.route_logits[:, :1], 50.0), self.output.route_logits[:, 1:]), dim=1),
            amount_logits=self.output.amount_logits,
            value=self.output.value,
        )
        self.assertLess(entropy_parts(certain)[1].max().item(), 1e-6)

    def test_base_order_does_not_matter(self):
        generator = torch.Generator().manual_seed(1)
        for state, legal in zip(self.states[:8], self.legal[:8]):
            count = len(state.base_ids)
            order = torch.randperm(count, generator=generator).tolist()
            with torch.no_grad():
                original = self.model(make_input([state], [legal]))
                shuffled = self.model(make_input(*[[x] for x in reordered(state, legal, order)]))

            self.assertTrue(torch.allclose(shuffled.value, original.value, atol=1e-5))
            routes = original.route_logits[0, 1:].view(count, count)[order][:, order]
            self.assertTrue(torch.allclose(shuffled.route_logits[0, 1:].view(count, count), routes, atol=1e-4))
            self.assertTrue(torch.allclose(shuffled.route_logits[0, 0], original.route_logits[0, 0], atol=1e-5))
            amounts = original.amount_logits[0][order][:, order]
            self.assertTrue(torch.allclose(shuffled.amount_logits[0], amounts, atol=1e-4))

    def test_padding_does_not_change_the_answer(self):
        sizes = [len(state.base_ids) for state in self.states]
        small = sizes.index(min(sizes))
        self.assertLess(sizes[small], max(sizes))
        count = sizes[small]
        with torch.no_grad():
            alone = self.model(make_input([self.states[small]], [self.legal[small]]))
        padded = self.inputs.base_mask.shape[1]

        self.assertTrue(torch.allclose(self.output.value[small], alone.value[0], atol=1e-5))
        in_batch = self.output.route_logits[small, 1:].view(padded, padded)[:count, :count]
        self.assertTrue(torch.allclose(in_batch, alone.route_logits[0, 1:].view(count, count), atol=1e-4))
        self.assertTrue(
            torch.allclose(self.output.amount_logits[small, :count, :count], alone.amount_logits[0], atol=1e-4)
        )

    def test_every_weight_can_learn(self):
        model = build_model()
        output = model(self.inputs)
        route, amount = choose(output)
        loss = log_probability(output, route, amount).sum() + entropy(output).sum() + output.value.sum()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum(), 0, name)

    def test_a_model_built_before_options_were_added_never_picks_them(self):
        older = build_model(amount_options=len(AMOUNT_OPTIONS) - 2).eval()
        with torch.no_grad():
            output = older(self.inputs)
        self.assertEqual(output.amount_logits.shape[-1], len(AMOUNT_OPTIONS))
        # On every usable route the options it does not know get nothing.
        probability = torch.softmax(output.amount_logits, dim=-1)[self.inputs.allowed.any(dim=-1)]
        self.assertEqual(probability[:, -2:].max().item(), 0.0)
        # Every route that is usable at all is still usable, through the options it knows.
        usable = self.inputs.allowed.any(dim=-1).flatten(start_dim=1)
        self.assertTrue((output.route_logits[:, 1:] > -1e8).eq(usable).all())
        for _ in range(5):
            _, amount = choose(output)
            self.assertTrue((amount < len(AMOUNT_OPTIONS) - 2).all())

    def test_the_capture_output_scores_every_route_and_amount(self):
        model = build_model(capture_head=True).eval()
        with torch.no_grad():
            output = model(self.inputs)
        self.assertEqual(output.capture_logits.shape, output.amount_logits.shape)
        self.assertIsNone(self.output.capture_logits)  # off unless asked for
        # Its weights are extra: the rest of the network is the same size.
        plain = sum(p.numel() for p in build_model().parameters())
        extra = sum(p.numel() for p in model.capture_out.parameters())
        self.assertEqual(sum(p.numel() for p in model.parameters()), plain + extra)

    def test_the_usable_route_list_is_padded_to_a_few_lengths(self):
        lengths = set()
        for count in range(0, 5000, 7):
            usable = torch.zeros(40, 30, 30, dtype=torch.bool)
            usable.view(-1)[:count] = True
            position, source, target = padded_route_list(usable, batch=40)
            self.assertGreaterEqual(len(position), count)
            self.assertLess(len(position), count + max(8, count / 4))
            # Real entries first, in order; the padding points at the extra position only.
            real = usable.nonzero(as_tuple=True)
            self.assertTrue(torch.equal(position[:count], real[0]))
            self.assertTrue((position[count:] == 40).all())
            lengths.add(len(position))
        # Far fewer different lengths than different route counts.
        self.assertLess(len(lengths), 60)

    def test_any_architecture_works_through_the_same_interface(self):
        model = EverythingEqualModel()
        output = model(self.inputs)
        bases = self.inputs.base_mask.shape[1]
        route, amount = choose(output)
        for choice, legal in zip(to_choices(route, amount, bases), self.legal):
            decode_action(choice, legal)

        # With no preferences, every legal route and waiting are equally likely.
        routes = self.inputs.allowed.any(dim=-1).flatten(1).sum(dim=1)
        waiting = torch.softmax(output.route_logits, dim=1)[:, 0]
        self.assertTrue(torch.allclose(waiting, 1 / (1 + routes), atol=1e-6))
        self.assertTrue(torch.isfinite(entropy(output)).all())


class FeatureSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        states, legal = positions(seeds=range(3), every=50, ticks=200)
        cls.inputs = make_input(states, legal)
        torch.manual_seed(0)
        cls.model = build_model(feature_set="v2", amounts="free").eval()

    def output(self, inputs):
        with torch.no_grad():
            return self.model(inputs)

    def changed(self, table, columns):
        values = getattr(self.inputs, table).clone()
        values[..., list(columns)] += 5.0
        return replace(self.inputs, **{table: values})

    def test_the_sizes_follow_the_set(self):
        config = self.model.config
        self.assertEqual((config.base_features, config.pair_features, config.route_features, config.global_features), (34, 3, 2, 4))

    def test_columns_outside_the_set_change_nothing(self):
        columns = feature_columns("v2")
        original = self.output(self.inputs)
        for table, every in (("base_features", BASE_FEATURES), ("route_features", ROUTE_FEATURES)):
            unread = [column for column in range(len(every)) if column not in columns[table]]
            self.assertTrue(unread)
            other = self.output(self.changed(table, unread))
            self.assertTrue(torch.equal(original.route_logits, other.route_logits), table)
            self.assertTrue(torch.equal(original.amount_curves, other.amount_curves), table)
            self.assertTrue(torch.equal(original.value, other.value), table)

    def test_each_column_in_the_set_is_read(self):
        original = self.output(self.inputs)
        for table, columns in feature_columns("v2").items():
            for column in columns:
                other = self.output(self.changed(table, [column]))
                self.assertFalse(torch.equal(original.value, other.value), (table, column))

    def test_an_unknown_set_is_refused(self):
        with self.assertRaises(ValueError):
            build_model(feature_set="v9")


if __name__ == "__main__":
    unittest.main()
