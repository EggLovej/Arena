from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType

from ricochet.api import Controller
from ricochet.resources import contributor_directory


def default_ais_directory() -> Path:
    return contributor_directory("ais")


def available_ais(directory: Path | None = None) -> tuple[str, ...]:
    root = directory or default_ais_directory()
    return tuple(sorted(path.stem for path in root.glob("*.py") if not path.name.startswith("_")))


def _load_module(path: Path) -> ModuleType:
    module_name = f"ricochet_external_ai_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load AI module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_controller(name: str, directory: Path | None = None) -> Controller:
    if name.startswith("checkpoint:"):
        from ricochet.training.checkpoints import CheckpointController

        return CheckpointController(name.split(":", 1)[1])
    root = directory or default_ais_directory()
    module_name = name.removesuffix(".py")
    if not module_name.isidentifier() or module_name.startswith("_"):
        raise ValueError("AI names must be public Python module names")
    path = root / f"{module_name}.py"
    if not path.is_file():
        raise ValueError(f"unknown AI '{module_name}'; available: {', '.join(available_ais(root))}")
    module = _load_module(path)
    factory = getattr(module, "create_controller", None)
    if not callable(factory):
        raise ValueError(f"{path.name} must define create_controller()")
    controller = factory()
    if not callable(getattr(controller, "reset", None)) or not callable(
        getattr(controller, "act", None)
    ):
        raise ValueError(f"{path.name} controller must define reset(context) and act(observation)")
    return controller


def controller_label(name: str) -> str:
    if name.startswith("checkpoint:"):
        return "Neural: " + Path(name.split(":", 1)[1]).parent.name
    return name.removesuffix(".py").removesuffix("_ai").replace("_", " ").title()
