"""The numbers behind the rules. Lengths are in pixels, times in seconds, energy in points."""

import math
from dataclasses import asdict, dataclass

RULES_VERSION = 2  # Stored in replays and checkpoints, which refuse any other version.


@dataclass(frozen=True)
class RulesConfig:
    """Every number the rules use; the defaults are the rules in RULES.md."""

    shield_activation_cost: float = 16.0  # energy to raise the shield, besides its prepaid minimum
    shield_min_seconds: float = 0.2  # a raised shield stays up at least this long
    ticks_per_second: int = 30
    max_seconds: float = 90.0  # match length; reaching it is a draw
    starting_health: int = 3  # each hit takes one
    max_energy: float = 100.0  # energy points; tanks start full
    energy_regen_per_second: float = 12.0
    movement_energy_per_second: float = 16.0  # at full speed, less in proportion when slower
    shield_energy_per_second: float = 36.0  # while the shield is up
    shield_speed_multiplier: float = 0.6  # share of tank_speed while shielded
    fire_energy: float = 32.0  # per shot
    fire_cooldown_seconds: float = 0.5  # between two shots of one tank
    tank_speed: float = 180.0  # pixels per second
    tank_radius: float = 18.0  # pixels
    projectile_speed: float = 420.0  # pixels per second
    projectile_radius: float = 4.0  # pixels
    projectile_lifetime: float = 5.0  # seconds before an unspent shot disappears
    max_projectiles: int = 24  # shots in flight at once; nobody fires beyond it
    max_walls: int = 12  # per map
    max_bot_faults: int = 3  # standalone game only: faults before a bot forfeits

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name not in {"shield_activation_cost", "shield_min_seconds"} and value == 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "ticks_per_second",
            "starting_health",
            "max_projectiles",
            "max_walls",
            "max_bot_faults",
        ):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.max_ticks < 1 or self.shield_speed_multiplier > 1:
            raise ValueError("invalid duration or shield speed multiplier")

    @property
    def shield_min_ticks(self) -> int:
        return max(1, math.ceil(self.shield_min_seconds * self.ticks_per_second))

    @property
    def dt(self) -> float:
        return 1.0 / self.ticks_per_second

    @property
    def max_ticks(self) -> int:
        return round(self.max_seconds * self.ticks_per_second)

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, int | float]) -> "RulesConfig":
        return cls(**values)
