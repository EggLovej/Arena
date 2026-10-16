"""Deterministic game rules with no dependency on Pygame."""

from dataclasses import dataclass

from game_types import (
    BLUE,
    NEUTRAL,
    PLAYERS,
    BaseView,
    EdgeView,
    FleetView,
    GameView,
    MapDefinition,
    SendAction,
)


@dataclass
class Base:
    id: str
    owner: int
    units: float
    production_rate: float
    x: int
    y: int


@dataclass
class Fleet:
    owner: int
    units: int
    source_id: str
    target_id: str
    remaining_time: float
    total_time: float


class Game:
    """Owns mutable match state and applies one simultaneous game tick."""

    def __init__(self, map_definition: MapDefinition, seconds_per_tick: float = 0.1):
        if seconds_per_tick <= 0:
            raise ValueError("seconds_per_tick must be positive")

        self.map_definition = map_definition
        self.seconds_per_tick = seconds_per_tick
        self.tick = 0
        self.time_seconds = 0.0
        starting_owners = {base.owner for base in map_definition.bases}
        self.players = tuple(player for player in PLAYERS if player in starting_owners)

        self.bases = {
            definition.id: Base(
                id=definition.id,
                owner=definition.owner,
                units=definition.units,
                production_rate=definition.production_rate,
                x=definition.x,
                y=definition.y,
            )
            for definition in map_definition.bases
        }
        self.edges = map_definition.edges
        self.fleets: list[Fleet] = []

        self._travel_times = {}
        for edge in self.edges:
            self._travel_times[frozenset((edge.a, edge.b))] = edge.travel_time

    def view_for(self, player_id: int) -> GameView:
        if player_id not in self.players:
            raise ValueError(f"Unknown player: {player_id}")

        bases = tuple(
            BaseView(
                id=base.id,
                owner=base.owner,
                units=base.units,
                production_rate=base.production_rate,
                x=base.x,
                y=base.y,
            )
            for base in self.bases.values()
        )
        edges = tuple(
            EdgeView(a=edge.a, b=edge.b, travel_time=edge.travel_time)
            for edge in self.edges
        )
        fleets = tuple(
            FleetView(
                owner=fleet.owner,
                units=fleet.units,
                source_id=fleet.source_id,
                target_id=fleet.target_id,
                remaining_time=fleet.remaining_time,
                total_time=fleet.total_time,
            )
            for fleet in self.fleets
        )
        return GameView(
            player_id=player_id,
            tick=self.tick,
            time_seconds=self.time_seconds,
            seconds_per_tick=self.seconds_per_tick,
            bases=bases,
            edges=edges,
            fleets=fleets,
        )

    def are_connected(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self._travel_times

    def step(self, actions: dict[int, SendAction | None]) -> dict[int, str]:
        """Advance one tick and return explanations for rejected actions."""

        rejected_actions = {}
        valid_actions = []

        # Every action is checked against the same pre-tick state.
        for player_id in self.players:
            action = actions.get(player_id)
            error = self._validate_action(player_id, action)
            if error is not None:
                rejected_actions[player_id] = error
            elif action is not None:
                valid_actions.append((player_id, action))

        # Players cannot share a source, so validated sends can start together.
        for player_id, action in valid_actions:
            source = self.bases[action.source_id]
            travel_time = self._travel_times[
                frozenset((action.source_id, action.target_id))
            ]
            source.units -= action.units
            self.fleets.append(
                Fleet(
                    owner=player_id,
                    units=action.units,
                    source_id=action.source_id,
                    target_id=action.target_id,
                    remaining_time=travel_time,
                    total_time=travel_time,
                )
            )

        self._produce_units()
        arrivals = self._move_fleets()
        self._resolve_arrivals(arrivals)

        self.tick += 1
        self.time_seconds = self.tick * self.seconds_per_tick
        return rejected_actions

    def _validate_action(self, player_id: int, action: SendAction | None) -> str | None:
        if action is None:
            return None
        if not isinstance(action, SendAction):
            return "The AI must return SendAction or None"
        if action.source_id not in self.bases:
            return f"Unknown source base: {action.source_id}"
        if action.target_id not in self.bases:
            return f"Unknown target base: {action.target_id}"
        if action.source_id == action.target_id:
            return "Source and target must be different"
        if not isinstance(action.units, int) or isinstance(action.units, bool):
            return "Units must be an integer"
        if action.units <= 0:
            return "Units must be positive"

        source = self.bases[action.source_id]
        if source.owner != player_id:
            return "The source base is not owned by this player"
        if action.units > int(source.units):
            return "The source base does not have enough units"

        if not self.are_connected(action.source_id, action.target_id):
            return "Source and target are not connected"
        return None

    def _produce_units(self) -> None:
        for base in self.bases.values():
            if base.owner in PLAYERS:
                base.units += base.production_rate * self.seconds_per_tick

    def _move_fleets(self) -> dict[str, list[Fleet]]:
        arrivals: dict[str, list[Fleet]] = {}
        travelling_fleets = []

        for fleet in self.fleets:
            fleet.remaining_time -= self.seconds_per_tick
            if fleet.remaining_time <= 1e-9:
                arrivals.setdefault(fleet.target_id, []).append(fleet)
            else:
                travelling_fleets.append(fleet)

        self.fleets = travelling_fleets
        return arrivals

    def _resolve_arrivals(self, arrivals: dict[str, list[Fleet]]) -> None:
        for target_id, arriving_fleets in arrivals.items():
            target = self.bases[target_id]
            forces = {}

            if target.units > 1e-9:
                forces[target.owner] = target.units
            for fleet in arriving_fleets:
                forces[fleet.owner] = forces.get(fleet.owner, 0.0) + fleet.units

            ranked_forces = sorted(forces.items(), key=lambda item: item[1], reverse=True)
            if not ranked_forces:
                target.owner = NEUTRAL
                target.units = 0.0
                continue

            strongest_owner, strongest_units = ranked_forces[0]
            second_units = ranked_forces[1][1] if len(ranked_forces) > 1 else 0.0

            if abs(strongest_units - second_units) <= 1e-9:
                target.owner = NEUTRAL
                target.units = 0.0
            else:
                target.owner = strongest_owner
                target.units = strongest_units - second_units

    def living_players(self) -> tuple[int, ...]:
        living = set()
        for base in self.bases.values():
            if base.owner in PLAYERS:
                living.add(base.owner)
        for fleet in self.fleets:
            if fleet.units > 0:
                living.add(fleet.owner)
        return tuple(player for player in self.players if player in living)

    def is_finished(self) -> bool:
        return len(self.living_players()) <= 1

    def winner(self) -> int | None:
        living = self.living_players()
        return living[0] if len(living) == 1 else None

    def fleet_location(self, fleet: Fleet) -> tuple[float, float]:
        source = self.bases[fleet.source_id]
        target = self.bases[fleet.target_id]
        progress = 1.0 - fleet.remaining_time / fleet.total_time
        x = source.x + (target.x - source.x) * progress
        y = source.y + (target.y - source.y) * progress
        return x, y
