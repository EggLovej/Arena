"""Model definitions. Every architecture here can be swapped for any other.

To add an architecture:

1. Create a file in this folder with a config dataclass and a subclass of
   ``PolicyValueModel`` that implements ``scores``.
2. Add one line to ``MODELS`` below.

It can then be selected by name wherever a model is built.
"""

from ml.models.graph_transformer import GraphTransformer, GraphTransformerConfig
from ml.models.interface import ModelInput, ModelOutput, PolicyValueModel, Scores, make_input


# name -> (model class, config class)
MODELS = {
    "graph_transformer": (GraphTransformer, GraphTransformerConfig),
}
DEFAULT_MODEL = "graph_transformer"


def build_model(name: str = DEFAULT_MODEL, **settings) -> PolicyValueModel:
    """Create an untrained model, for example ``build_model("graph_transformer", width=96)``."""

    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available models: {', '.join(MODELS)}")
    model_class, config_class = MODELS[name]
    return model_class(config_class(**settings))


def name_of(model: PolicyValueModel) -> str:
    """The name a model was built under, needed to rebuild it from a saved file."""

    for name, (model_class, _) in MODELS.items():
        if type(model) is model_class:
            return name
    raise ValueError(f"{type(model).__name__} is not listed in MODELS")


__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "ModelInput",
    "ModelOutput",
    "PolicyValueModel",
    "Scores",
    "build_model",
    "make_input",
    "name_of",
]
