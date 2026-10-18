import math
from dataclasses import asdict, dataclass

RULES_VERSION = 2


@dataclass(frozen=True)
class RulesConfig:
    shield_activation_cost: float = 16.0
    shield_min_seconds: float = 0.2
    ticks_per_second: int = 30
    max_seconds: float = 90.0
    starting_health: int = 3
    max_energy: float = 100.0
    energy_regen_per_second: float = 12.0
    movement_energy_per_second: float = 16.0
    shield_energy_per_second: float = 36.0
    shield_speed_multiplier: float = 0.6
    fire_energy: float = 32.0
    fire_cooldown_seconds: float = 0.5
    tank_speed: float = 180.0
    tank_radius: float = 18.0
    projectile_speed: float = 420.0
    projectile_radius: float = 4.0
    projectile_lifetime: float = 5.0
    max_projectiles: int = 24
    max_walls: int = 12
    max_bot_faults: int = 3

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
