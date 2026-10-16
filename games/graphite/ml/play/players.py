"""Players built on a network, by name: ``value:PATH``, ``lookahead:PATH`` and ``network:PATH``.

Wherever a player is named (the evaluator, the rollouts, and so every
trainer), such a name turns a checkpoint into a player: one that thinks before
it moves (``value:``, ``lookahead:``), or the network on its own at a chosen
temperature (``network:``). Settings can follow the path:
``lookahead:PATH?mine=8&rule=nash&judge=PATH``, ``network:PATH?temperature=0.2``.
"""

from dataclasses import fields


def _kinds() -> dict:
    """Name prefix -> (settings class, a function building the player from a path, settings and seed)."""

    from ml.play.agent import PREFIX as NETWORK, NetworkSettings, NeuralAgent
    from ml.play.lookahead import PREFIX as LOOKAHEAD, LookaheadPlayer, LookaheadSettings
    from ml.play.value_player import PREFIX as VALUE, ValuePlayer, ValueSettings

    return {
        VALUE: (ValueSettings, lambda path, settings, seed: ValuePlayer.from_init(path, settings)),
        LOOKAHEAD: (LookaheadSettings, lambda path, settings, seed: LookaheadPlayer.from_init(path, settings)),
        NETWORK: (
            NetworkSettings,
            lambda path, settings, seed: NeuralAgent.from_init(path, seed=seed, temperature=settings.temperature),
        ),
    }


def is_network_player(name: str) -> bool:
    return name.startswith(tuple(_kinds()))


def network_player(name: str, seed: int | None = None):
    """The player a ``value:``, ``lookahead:`` or ``network:`` name stands for."""

    for prefix, (settings_class, build) in _kinds().items():
        if name.startswith(prefix):
            path, settings = parse_name(name.removeprefix(prefix), settings_class)
            return build(path, settings, seed)
    raise ValueError(f"'{name}' is not a value:, lookahead: or network: player")


def parse_name(text: str, settings_class):
    """``PATH?key=value&...`` as the path and a settings object, each value read as its setting's type."""

    path, _, options = text.partition("?")
    defaults = {setting.name: setting.default for setting in fields(settings_class)}
    changes = {}
    for option in filter(None, options.split("&")):
        key, _, value = option.partition("=")
        if key not in defaults:
            raise ValueError(f"'{key}' is not a setting here. Settings: {', '.join(defaults)}")
        changes[key] = type(defaults[key])(value)
    return path, settings_class(**changes)


def name_with_settings(prefix: str, path, settings) -> str:
    """The player name for this checkpoint and these settings; settings at their default are left out."""

    defaults = type(settings)()
    changed = [
        f"{setting.name}={getattr(settings, setting.name)}"
        for setting in fields(settings)
        if getattr(settings, setting.name) != getattr(defaults, setting.name)
    ]
    return f"{prefix}{path}" + ("?" + "&".join(changed) if changed else "")
