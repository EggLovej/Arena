import unittest
from dataclasses import replace

import torch

from ml.encoding.action_encoder import AMOUNT_OPTIONS, LegalActions, decode_action
from ml.encoding.state_encoder import BASE_FEATURES, ROUTE_FEATURES, EncodedState, feature_columns
from ml.models import build_model, make_input
from ml.models.interface import ILLEGAL, ModelInput, ModelOutput, PolicyValueModel, Scores, padded_route_list
from ml.models.policy import choose, entropy, entropy_parts, log_probability, to_choices
from ml.tests.support import positions, repeated


def every_choice(inputs: ModelInput, output: ModelOutput, row: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every choice a network has in one position, and its chance of making it: waiting, then each usable route
    with each amount it may send (with free amounts any number of units up to the most, else the allowed options)."""

    bases = inputs.base_mask.shape[1]
    routes, amounts = [torch.zeros(1, dtype=torch.long)], [torch.zeros(1, dtype=torch.long)]
    for source, target in inputs.allowed[row].any(dim=-1).nonzero().tolist():
        if output.amount_curves is not None:
            each = torch.arange(1, int(inputs.most[row, source, target]) + 1)
        else:
            each = inputs.allowed[row, source, target].nonzero()[:, 0]
        routes.append(torch.full((len(each),), 1 + source * bases + target))
        amounts.append(each)
    routes, amounts = torch.cat(routes), torch.cat(amounts)
    return routes, amounts, log_probability(repeated(output, row, len(routes)), routes, amounts).exp()


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


class ModelTests(unittest.TestCase):
    """The rules every network keeps, with amount options and with free amounts."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.states, cls.legal = positions()
        cls.inputs = make_input(cls.states, cls.legal)
        cls.models = {amounts: build_model(amounts=amounts).eval() for amounts in ("options", "free")}
        with torch.no_grad():
            cls.outputs = {amounts: model(cls.inputs) for amounts, model in cls.models.items()}
        cls.model, cls.output = cls.models["options"], cls.outputs["options"]

    def test_untrained_model_is_undecided(self):
        self.assertLess(self.output.value.abs().max(), 0.2)
        route_probability = torch.softmax(self.output.route_logits, dim=1)
        routes = self.inputs.allowed.any(dim=-1).flatten(1).sum(dim=1)
        # Waiting is roughly one choice among all the legal routes.
        expected = 1 / (1 + routes)
        self.assertTrue(torch.allclose(route_probability[:, 0], expected, rtol=0.3))

    def test_illegal_choices_have_zero_probability(self):
        # A route is usable when some amount may go along it, whichever kind of amounts.
        usable = self.inputs.allowed.any(dim=-1)
        self.assertTrue(torch.equal(usable, self.inputs.most > 0))
        for amounts, output in self.outputs.items():
            self.assertEqual(torch.softmax(output.route_logits, dim=1)[:, 1:][~usable.flatten(1)].max(), 0.0, amounts)
        amount_probability = torch.softmax(self.output.amount_logits, dim=-1)[usable]
        self.assertEqual(amount_probability[~self.inputs.allowed[usable]].max(), 0.0)

    def test_every_sampled_choice_is_a_legal_move(self):
        bases = self.inputs.base_mask.shape[1]
        for amounts, output in self.outputs.items():
            for _ in range(20):
                route, amount = choose(output)
                for choice, legal in zip(to_choices(route, amount, bases), self.legal):
                    decode_action(choice, legal, free=amounts == "free")  # raises if the choice is not legal

    def test_greedy_takes_the_likeliest_route_then_its_likeliest_amount(self):
        # Not always the likeliest pair: waiting has no amount to split its chance over.
        for amounts, output in self.outputs.items():
            route, amount = choose(output, greedy=True)
            self.assertTrue(torch.equal(route, output.route_logits.argmax(dim=1)), amounts)
            for row in (route > 0).nonzero()[:, 0].tolist():
                routes, options, chance = every_choice(self.inputs, output, row)
                on_route = routes == route[row]
                self.assertEqual(amount[row], options[on_route][chance[on_route].argmax()], amounts)

    def test_a_temperature_sharpens_the_odds_of_the_route_and_the_amount(self):
        count = 4000
        # Routes: clearly different odds for the usable routes of the first position, so sharpening shows.
        logits = self.output.route_logits[0].clone()
        usable = logits > ILLEGAL
        logits[usable] = torch.linspace(0.0, 2.0, int(usable.sum()))
        many = replace(repeated(self.output, 0, count), route_logits=logits.expand(count, -1))
        plain = choose(many, generator=torch.Generator().manual_seed(1))
        at_one = choose(many, generator=torch.Generator().manual_seed(1), temperature=1.0)
        self.assertTrue(torch.equal(plain[0], at_one[0]) and torch.equal(plain[1], at_one[1]))
        route, _ = choose(many, generator=torch.Generator().manual_seed(1), temperature=0.5)
        expected = torch.softmax(logits * 2, dim=0)
        counts = torch.bincount(route, minlength=len(expected)).float() / count
        self.assertTrue(((counts - expected).abs() <= 5 * (expected * (1 - expected) / count).sqrt() + 1e-3).all())
        self.assertGreater(counts[int(logits.argmax())], (plain[0] == logits.argmax()).float().mean() + 0.05)
        with self.assertRaises(ValueError):
            choose(self.output, temperature=0)

        # Amounts: one route only, the one with the most units to send, for each kind of amounts.
        chosen = 1 + int(self.inputs.most[0].argmax())
        only = torch.full_like(logits, ILLEGAL)
        only[chosen] = 0.0
        for amounts, output in self.outputs.items():
            many = replace(repeated(output, 0, count), route_logits=only.expand(count, -1))
            if amounts == "options":  # clearly different odds for the options too
                amount_logits = many.amount_logits[0].clone()
                source, target = divmod(chosen - 1, self.inputs.base_mask.shape[1])
                legal = amount_logits[source, target] > ILLEGAL
                amount_logits[source, target, legal] = torch.linspace(0.0, 2.0, int(legal.sum()))
                many = replace(many, amount_logits=amount_logits.expand(count, *amount_logits.shape))
            _, drawn = choose(many, generator=torch.Generator().manual_seed(2), temperature=0.5)
            routes, options, chance = every_choice(self.inputs, many, 0)
            options, chance = options[routes == chosen], chance[routes == chosen]
            self.assertGreater(len(options), 2)
            expected = chance**2 / (chance**2).sum()
            counts = torch.stack([(drawn == option).float().mean() for option in options])
            spread = (expected * (1 - expected) / count).sqrt()
            self.assertTrue(((counts - expected).abs() <= 5 * spread + 1e-3).all(), amounts)

    def test_probabilities_of_all_choices_sum_to_one_and_entropy_agrees(self):
        for amounts, output in self.outputs.items():
            chances = [every_choice(self.inputs, output, row)[2] for row in range(len(self.states))]
            total = torch.stack([chance.sum() for chance in chances])
            self.assertTrue(torch.allclose(total, torch.ones(len(chances)), atol=1e-4), amounts)
            direct = torch.stack([-(chance * torch.log(chance.clamp(min=1e-30))).sum() for chance in chances])
            self.assertTrue(torch.allclose(entropy(output), direct, atol=1e-3), amounts)

    def test_entropy_parts_are_the_route_choice_and_the_rest(self):
        for amounts, output in self.outputs.items():
            route_part, amount_part = entropy_parts(output)
            route_probability = torch.softmax(output.route_logits, dim=1)
            direct = -(route_probability * torch.log(route_probability.clamp(min=1e-30))).sum(dim=1)
            self.assertTrue(torch.allclose(route_part, direct, atol=1e-5), amounts)
            self.assertTrue(torch.allclose(route_part + amount_part, entropy(output)), amounts)
            # A position where only waiting is likely has (almost) no amount part.
            waiting = torch.full_like(output.route_logits[:, :1], 50.0)
            certain = replace(output, route_logits=torch.cat((waiting, output.route_logits[:, 1:]), dim=1))
            self.assertLess(entropy_parts(certain)[1].max().item(), 1e-6, amounts)

    def test_base_order_does_not_matter(self):
        generator = torch.Generator().manual_seed(1)
        for state, legal in zip(self.states[:8], self.legal[:8]):
            count = len(state.base_ids)
            order = torch.randperm(count, generator=generator).tolist()
            for model in self.models.values():
                with torch.no_grad():
                    original = model(make_input([state], [legal]))
                    shuffled = model(make_input(*[[x] for x in reordered(state, legal, order)]))
                self.assertTrue(torch.allclose(shuffled.value, original.value, atol=1e-5))
                routes = original.route_logits[0, 1:].view(count, count)[order][:, order]
                self.assertTrue(torch.allclose(shuffled.route_logits[0, 1:].view(count, count), routes, atol=1e-4))
                self.assertTrue(torch.allclose(shuffled.route_logits[0, 0], original.route_logits[0, 0], atol=1e-5))
                for name in ("amount_logits", "amount_curves", "most", "needed"):
                    if getattr(original, name) is not None:
                        moved = getattr(original, name)[0][order][:, order]
                        self.assertTrue(torch.allclose(getattr(shuffled, name)[0], moved, atol=1e-4), name)

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
        for amounts in self.models:
            model = build_model(amounts=amounts)
            output = model(self.inputs)
            route, amount = choose(output)
            loss = log_probability(output, route, amount).sum() + entropy(output).sum() + output.value.sum()
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (amounts, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), (amounts, name))
                self.assertGreater(parameter.grad.abs().sum(), 0, (amounts, name))

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

    def test_settings_that_make_no_network_are_refused(self):
        for settings in (
            {"name": "no_such_model"},
            {"amounts": "any"},
            {"route_layers": 0},
            {"feature_set": "v9"},
        ):
            with self.subTest(**settings), self.assertRaises(ValueError):
                build_model(**settings)

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
        states, legal = positions(seeds=range(3), every=50)
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


if __name__ == "__main__":
    unittest.main()
