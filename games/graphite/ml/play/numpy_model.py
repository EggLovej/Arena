"""The graph transformer worked out with numpy alone, for playing where PyTorch is not installed.

The website plays bots in the visitor's browser, which runs Python but not
PyTorch, and the game itself can play the learned bots without it too. This
file repeats, for one position at a time and without learning, exactly what
these do:

- ``GraphTransformer.scores`` (ml/models/graph_transformer.py) and
  ``PolicyValueModel.forward`` (ml/models/interface.py): the scores of every
  choice, illegal ones removed
- ``choose`` (ml/models/policy.py) and its amount maths (ml/models/amounts.py):
  one choice drawn from them

A checkpoint is read straight from the file ``save_checkpoint`` writes,
without PyTorch (``read_checkpoint``). ml/tests/test_numpy_model.py checks
that both give the same scores for the committed networks and for every
setting of the architecture, so a change to the network in one place and not
the other fails there.
"""

import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.encoding.action_encoder import AMOUNT_OPTIONS, JUST_ENOUGH, LegalActions, SendChoice
from ml.encoding.state_encoder import (
    BASE_FEATURES,
    COLUMNS_BEFORE_V2,
    GLOBAL_FEATURES,
    PAIR_FEATURES,
    ROUTE_FEATURES,
    EncodedState,
    feature_columns,
)


FORMAT_VERSION = 1  # as ml/models/checkpoint.py
ILLEGAL = -1e9  # as ml/models/interface.py
LAYER_NORM_EPSILON = 1e-5  # PyTorch's default
CURVES = 3  # as ml/models/amounts.py
PARAMETERS = 5
MIN_WIDTH = 0.05

# The settings of GraphTransformerConfig and their defaults; a checkpoint stores the ones it changed.
DEFAULTS = {
    "width": 64,
    "layers": 3,
    "heads": 4,
    "route_layers": 1,
    "base_features": COLUMNS_BEFORE_V2["base_features"],
    "pair_features": len(PAIR_FEATURES),
    "route_features": COLUMNS_BEFORE_V2["route_features"],
    "global_features": len(GLOBAL_FEATURES),
    "amount_options": len(AMOUNT_OPTIONS),
    "capture_head": False,
    "amounts": "options",
    "amount_curves": CURVES,
    "feature_set": "",
}


# ---- Reading a checkpoint without PyTorch -------------------------------------------------

def read_checkpoint(path: str | Path) -> dict:
    """What ``save_checkpoint`` wrote, with every tensor as a numpy array.

    ``torch.save`` writes a zip file: a pickle of the contents, where each tensor
    points at its numbers, stored raw in a file of their own. Only tensors and plain
    Python values are read, as ``torch.load(weights_only=True)`` does, so a file
    holding code is refused.
    """

    with zipfile.ZipFile(path) as archive:
        pickled = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        folder = pickled.removesuffix("data.pkl")
        if folder + "byteorder" in archive.namelist() and archive.read(folder + "byteorder") != b"little":
            raise ValueError(f"{path} was saved on a big-endian machine, which is not supported here")

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                    return _rebuild_tensor
                if (module, name) == ("collections", "OrderedDict"):
                    return OrderedDict
                if module == "torch" and name in _STORAGE_TYPES:
                    return _STORAGE_TYPES[name]
                raise pickle.UnpicklingError(f"{path} holds {module}.{name}, which a checkpoint never does")

            def persistent_load(self, saved_id):
                kind, dtype, key, _location, count = saved_id
                if kind != "storage":
                    raise pickle.UnpicklingError(f"{path} holds an unknown kind of data, {kind!r}")
                return np.frombuffer(archive.read(f"{folder}data/{key}"), dtype=dtype, count=count)

        with archive.open(pickled) as file:
            contents = Unpickler(file).load()
    if contents.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} uses checkpoint format {contents.get('format_version')}, expected {FORMAT_VERSION}")
    return contents


def _rebuild_tensor(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    # np.ndarray refuses a shape, offset or strides that reach outside the storage.
    item = storage.itemsize
    view = np.ndarray(tuple(size), dtype=storage.dtype, buffer=storage, offset=offset * item, strides=tuple(step * item for step in stride))
    return view.copy()


_STORAGE_TYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


# ---- The network ------------------------------------------------------------------------

@dataclass(frozen=True)
class NumpyOutput:
    """``ModelOutput`` for one position (ml/models/interface.py), without the batch axis."""

    route_logits: np.ndarray  # [1 + bases * bases]: wait, then every route, source then target
    amount_logits: np.ndarray | None  # [bases, bases, amount options]; None with free amounts
    value: float
    amount_curves: np.ndarray | None = None  # [bases, bases, curves, PARAMETERS] with free amounts
    most: np.ndarray | None = None  # [bases, bases]
    needed: np.ndarray | None = None


class NumpyModel:
    """A saved graph transformer, ready to score positions."""

    def __init__(self, config: dict, weights: dict[str, np.ndarray]):
        self.config = {**DEFAULTS, **config}
        if self.config["feature_set"]:
            for table, columns in feature_columns(self.config["feature_set"]).items():
                self.config[table] = len(columns)
        self.columns = feature_columns(self.config["feature_set"]) if self.config["feature_set"] else None
        self.weights = {name: np.asarray(weight, dtype=np.float32) for name, weight in weights.items()}

    @property
    def free_amounts(self) -> bool:
        return self.config["amounts"] == "free"

    @classmethod
    def from_file(cls, path: str | Path) -> "NumpyModel":
        """Load a checkpoint, refusing one trained on other features than the encoders produce now."""

        contents = read_checkpoint(path)
        if contents["model_name"] != "graph_transformer":
            raise ValueError(f"{path} holds a '{contents['model_name']}' model; only graph_transformer plays without PyTorch")
        current = {
            "base_features": BASE_FEATURES,
            "pair_features": PAIR_FEATURES,
            "route_features": ROUTE_FEATURES,
            "global_features": GLOBAL_FEATURES,
            "amount_options": AMOUNT_OPTIONS,
        }
        saved = contents["encoding"]
        changed = [name for name, names in saved.items() if list(names) != list(current.get(name, ()))[: len(names)]]
        if changed:
            raise ValueError(f"{path} was trained with different {', '.join(changed)} than the encoders produce now")
        config = dict(contents["model_config"])
        # A table that did not exist when the model was saved is one it never reads.
        for table in current.keys() - saved.keys():
            if table in DEFAULTS:
                config.setdefault(table, 0)
        return cls(config, contents["weights"])

    def __call__(self, state: EncodedState, legal: LegalActions) -> NumpyOutput:
        config, weight = self.config, self.weights
        bases = len(state.base_ids)
        heads = config["heads"]
        allowed = legal.allowed
        usable = allowed.any(axis=-1)

        pairs = np.concatenate(
            (
                self._read(state.pair_features, "pair_features"),
                self._read(state.route_features, "route_features"),
                allowed[..., JUST_ENOUGH : JUST_ENOUGH + 1].astype(np.float32),
            ),
            axis=-1,
        )
        # One attention nudge per layer and head: [layers, heads, listener, speaker].
        bias = self._linear("pair_bias.2", _relu(self._linear("pair_bias.0", pairs)))
        bias = bias.reshape(bases, bases, config["layers"], heads).transpose(2, 3, 0, 1)

        described = self._linear("embed_bases", self._read(state.base_features, "base_features"))
        described = described + self._linear("embed_globals", self._read(state.global_features, "global_features"))
        for layer in range(config["layers"]):
            described = self._attention_block(f"blocks.{layer}", described, bias[layer])
        described = self._layer_norm("final_norm", described)

        # Only the usable routes are scored; every other entry stays 0 and is removed as illegal.
        source, target = np.nonzero(usable)
        judged = _relu(
            self._linear("as_source", described)[source]
            + self._linear("as_target", described)[target]
            + self._linear("as_pair", pairs[source, target])
        )
        for layer in range(config["route_layers"] - 1):
            judged = _relu(self._linear(f"route_thinking.{2 * layer}", judged))
        outputs = weight["route_out.bias"].shape[0]
        route = np.zeros((bases, bases, outputs), dtype=np.float32)
        route[source, target] = self._linear("route_out", judged)

        summary = described.mean(axis=0)
        wait = self._linear("wait_out.2", _relu(self._linear("wait_out.0", summary)))[0]
        value = float(np.tanh(self._linear("value_out.2", _relu(self._linear("value_out.0", summary)))[0]))
        route_logits = np.concatenate(([wait], np.where(usable, route[..., 0], ILLEGAL).reshape(-1))).astype(np.float32)

        if self.free_amounts:
            curves = route[..., 1:].reshape(bases, bases, config["amount_curves"], PARAMETERS)
            most, needed = (part.astype(np.float32) for part in (legal.most, legal.needed))
            return NumpyOutput(route_logits, None, value, curves, most, needed)
        amount = route[..., 1:]
        known, options = amount.shape[-1], allowed.shape[-1]
        if known < options:
            # Options added after this network was built are never chosen.
            amount = np.concatenate((amount, np.full((bases, bases, options - known), ILLEGAL, np.float32)), axis=-1)
            allowed = allowed.copy()
            allowed[..., known:] = False
        return NumpyOutput(route_logits, np.where(allowed, amount, ILLEGAL).astype(np.float32), value)

    def _attention_block(self, name: str, described: np.ndarray, bias: np.ndarray) -> np.ndarray:
        bases, width = described.shape
        heads = self.config["heads"]
        size = width // heads
        parts = self._linear(f"{name}.query_key_value", self._layer_norm(f"{name}.norm_before_attention", described))
        query, key, value = parts.reshape(bases, 3, heads, size).transpose(1, 2, 0, 3)  # each [heads, bases, size]
        scores = query @ key.transpose(0, 2, 1) / np.sqrt(np.float32(size)) + bias
        heard = _softmax(scores) @ value
        described = described + self._linear(f"{name}.after_attention", heard.transpose(1, 0, 2).reshape(bases, width))
        thought = self._linear(f"{name}.thinking.0", self._layer_norm(f"{name}.norm_before_thinking", described))
        return described + self._linear(f"{name}.thinking.2", _relu(thought))

    def _linear(self, name: str, values: np.ndarray) -> np.ndarray:
        return values @ self.weights[f"{name}.weight"].T + self.weights[f"{name}.bias"]

    def _layer_norm(self, name: str, values: np.ndarray) -> np.ndarray:
        mean = values.mean(axis=-1, keepdims=True)
        variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
        normal = (values - mean) / np.sqrt(variance + LAYER_NORM_EPSILON)
        return normal * self.weights[f"{name}.weight"] + self.weights[f"{name}.bias"]

    def _read(self, values: np.ndarray, table: str) -> np.ndarray:
        """The columns of one input table that this network reads."""

        if self.columns is not None:
            return values[..., list(self.columns[table])]
        return values[..., : self.config[table]]


# ---- Choosing ---------------------------------------------------------------------------

def choose(
    output: NumpyOutput,
    greedy: bool = False,
    generator: np.random.Generator | None = None,
    temperature: float = 1.0,
) -> SendChoice | None:
    """One choice, as ``ml/models/policy.choose`` makes it; None is waiting."""

    if temperature <= 0:
        raise ValueError(f"temperature must be above 0, not {temperature}")
    generator = generator or np.random.default_rng()
    sharpen = 1.0 / temperature
    column = _pick(output.route_logits * sharpen, greedy, generator)
    if column == 0:
        return None
    bases = len(output.most if output.most is not None else output.amount_logits)
    source, target = divmod(column - 1, bases)
    if output.amount_curves is not None:
        curves = output.amount_curves[source, target]
        most, needed = float(output.most[source, target]), float(output.needed[source, target])
        if greedy:
            amount = int(on_grid(curves, most, needed, max(int(most), 1)).argmax()) + 1
        elif sharpen != 1.0:
            probability = on_grid(curves, most, needed, max(int(most), 1)) ** sharpen
            amount = int(generator.choice(len(probability), p=probability / probability.sum())) + 1
        else:
            amount = _sample_amount(curves, most, needed, generator)
    else:
        amount = _pick(output.amount_logits[source, target] * sharpen, greedy, generator)
    return SendChoice(int(source), int(target), int(amount))


def on_grid(parameters: np.ndarray, most: float, needed: float, size: int) -> np.ndarray:
    """``amounts.on_grid`` for one route: the probabilities of sending 1, 2, ..., ``size`` units."""

    units = np.arange(1, size + 1, dtype=np.float64)
    log_weight, centre, width = _curves(parameters, most, needed)
    per_curve = _log_area(units[:, None], centre[None, :], width[None, :], min(most, size))
    probability = np.exp(_logsumexp(log_weight[None, :] + per_curve))
    return probability * (units <= most)


def _sample_amount(parameters: np.ndarray, most: float, needed: float, generator: np.random.Generator) -> int:
    """``amounts.sample``: a curve by its weight, then a point on it."""

    log_weight, centre, width = _curves(parameters, most, needed)
    curve = generator.choice(len(log_weight), p=np.exp(log_weight) / np.exp(log_weight).sum())
    uniform = min(max(generator.random(), 1e-7), 1 - 1e-7)
    point = centre[curve] + width[curve] * (np.log(uniform) - np.log1p(-uniform))  # a logistic draw
    return int(min(max(round(point), 1), max(most, 1)))


def _curves(parameters: np.ndarray, most: float, needed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameters = parameters.astype(np.float64)
    log_weight = parameters[:, 0] - _logsumexp(parameters[:, 0])
    fraction = 1 / (1 + np.exp(-parameters[:, 1]))
    share = 2 / (1 + np.exp(-parameters[:, 2]))
    centre = fraction * most + share * needed + parameters[:, 3]
    width = np.logaddexp(0, parameters[:, 4]) + MIN_WIDTH
    return log_weight, centre, width


def _log_area(units: np.ndarray, centre: np.ndarray, width: np.ndarray, most: float) -> np.ndarray:
    """``amounts._log_area``: one curve's area over a whole number of units, the edges taking the tails."""

    upper = (units + 0.5 - centre) / width
    lower = (units - 0.5 - centre) / width
    with np.errstate(divide="ignore", invalid="ignore"):
        area = _log_sigmoid_difference(upper, lower)
    area = np.where(units <= 1, _log_sigmoid(upper), area)
    area = np.where(units >= most, _log_sigmoid(-lower), area)
    return np.where((units <= 1) & (units >= most), 0.0, area)


def _log_sigmoid_difference(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    mirror = low > 0
    high, low = np.where(mirror, -low, high), np.where(mirror, -high, low)
    gap = np.minimum(_log_sigmoid(low) - _log_sigmoid(high), -1e-30)
    return _log_sigmoid(high) + np.where(gap > -0.693, np.log(-np.expm1(gap)), np.log1p(-np.exp(gap)))


def _log_sigmoid(values: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0, -values)


def _logsumexp(values: np.ndarray) -> np.ndarray:
    largest = values.max(axis=-1, keepdims=True)
    return (largest + np.log(np.exp(values - largest).sum(axis=-1, keepdims=True)))[..., 0]


def _pick(logits: np.ndarray, greedy: bool, generator: np.random.Generator) -> int:
    if greedy:
        return int(logits.argmax())
    probability = _softmax(logits.astype(np.float64))
    return int(generator.choice(len(probability), p=probability))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.exp(values - values.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0)
