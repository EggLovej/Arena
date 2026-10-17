"""Atlas-style Pygame visualization. It never changes game state."""

import math

import pygame

from game import Game
from game_types import BLUE, GREEN, NEUTRAL, RED
from sizing import base_radii
from theme import (
    BLUE as BLUE_COLOR,
    GREEN as GREEN_COLOR,
    INK,
    MUSTARD,
    MUTED,
    PAPER_DARK,
    PAPER_LIGHT,
    RED as RED_COLOR,
    WHITE,
    font,
    load_background,
    load_brush,
    owner_color,
)


HEADER_HEIGHT = 84
FOOTER_HEIGHT = 54


class Renderer:
    def __init__(self, game: Game, player_names: dict[int, str] | None = None):
        """``player_names`` labels each player in the header, for example {RED: "Strategic"}."""

        pygame.init()
        self.player_names = {RED: "Red", BLUE: "Blue", GREEN: "Green", **(player_names or {})}
        size = (game.map_definition.width, game.map_definition.height)
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Graphite")

        self.background = load_background(size)
        self.red_brush = load_brush("red-brush-banner.png", (190, 55))
        self.blue_brush = load_brush("blue-brush-banner.png", (190, 55))

        base_count = len(game.bases)
        self.dense_map = base_count >= 25
        if self.dense_map:
            name_size, production_size = 12, 13
        elif base_count >= 15:
            name_size, production_size = 14, 15
        else:
            name_size, production_size = 16, 17

        # Production rates never change, so each base keeps one size all match.
        self.base_radii = base_radii(game.bases.values())
        self._count_fonts: dict[int, pygame.font.Font] = {}
        self.name_font = font(name_size, bold=True)
        self.production_font = font(production_size, bold=True)
        self.small_font = font(14, bold=True)
        self.body_font = font(16)
        self.header_font = font(22, bold=True)
        self.score_font = font(30, bold=True)
        self.result_font = font(45, bold=True)
        self.button_font = font(20, bold=True)

    def draw(
        self,
        game: Game,
        selected_base_id: str | None = None,
        streams: dict[str, str] | None = None,
    ) -> None:
        """``streams`` maps the source of each running stream to its target; they are drawn on their routes."""

        self._draw_board(game, selected_base_id, streams)
        pygame.display.flip()

    def _draw_board(
        self,
        game: Game,
        selected_base_id: str | None = None,
        streams: dict[str, str] | None = None,
    ) -> None:
        self.screen.blit(self.background, (0, 0))
        hovered_base_id = self._hovered_base(game)
        self._draw_edges(game, selected_base_id, hovered_base_id)
        self._draw_streams(game, streams or {})
        self._draw_fleets(game)
        self._draw_bases(game, selected_base_id, hovered_base_id)
        self._draw_header(game)
        self._draw_footer()

    def _draw_edges(
        self,
        game: Game,
        selected_base_id: str | None,
        hovered_base_id: str | None,
    ) -> None:
        for edge in game.edges:
            a = game.bases[edge.a]
            b = game.bases[edge.b]
            selected = selected_base_id in (edge.a, edge.b)
            outer_color = MUSTARD if selected else INK
            inner_color = PAPER_LIGHT if selected else PAPER_DARK

            pygame.draw.line(self.screen, outer_color, (a.x, a.y), (b.x, b.y), 7)
            pygame.draw.line(self.screen, inner_color, (a.x, a.y), (b.x, b.y), 3)

            if selected:
                start, end = (a, b) if selected_base_id == edge.a else (b, a)
                self._draw_route_chevron(start.x, start.y, end.x, end.y, 0.38)
                self._draw_route_chevron(start.x, start.y, end.x, end.y, 0.66)

            focused = (
                selected_base_id in (edge.a, edge.b)
                or hovered_base_id in (edge.a, edge.b)
            )
            if not self.dense_map or focused:
                midpoint = ((a.x + b.x) // 2, (a.y + b.y) // 2)
                self._draw_travel_tag(f"{edge.travel_time:g}s", midpoint)

    def _draw_streams(self, game: Game, streams: dict[str, str]) -> None:
        """A running stream: its route in the owner's colour, with arrows moving along it."""

        moving = (pygame.time.get_ticks() / 1000 * 0.6) % 0.25
        for source_id, target_id in streams.items():
            source, target = game.bases[source_id], game.bases[target_id]
            color = owner_color(source.owner)
            pygame.draw.line(self.screen, color, (source.x, source.y), (target.x, target.y), 5)
            for step in (0.2, 0.45, 0.7):
                self._draw_route_chevron(source.x, source.y, target.x, target.y, step + moving, color)
            # The travel time stays readable on top of the stream.
            edge = next(edge for edge in game.edges if {edge.a, edge.b} == {source_id, target_id})
            self._draw_travel_tag(f"{edge.travel_time:g}s", ((source.x + target.x) // 2, (source.y + target.y) // 2))

    def _draw_route_chevron(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        progress: float,
        color: tuple[int, int, int] = MUSTARD,
    ) -> None:
        x = start_x + (end_x - start_x) * progress
        y = start_y + (end_y - start_y) * progress
        angle = math.atan2(end_y - start_y, end_x - start_x)
        direction = (math.cos(angle), math.sin(angle))
        normal = (-direction[1], direction[0])
        points = [
            (x + direction[0] * 8, y + direction[1] * 8),
            (x - direction[0] * 6 + normal[0] * 6, y - direction[1] * 6 + normal[1] * 6),
            (x - direction[0] * 6 - normal[0] * 6, y - direction[1] * 6 - normal[1] * 6),
        ]
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, INK, points, width=1)

    def _draw_travel_tag(self, label: str, center: tuple[int, int]) -> None:
        text = self.small_font.render(label, True, INK)
        rect = text.get_rect(center=center).inflate(14, 8)
        pygame.draw.rect(self.screen, PAPER_LIGHT, rect, border_radius=5)
        pygame.draw.rect(self.screen, INK, rect, width=2, border_radius=5)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _draw_bases(
        self,
        game: Game,
        selected_base_id: str | None,
        hovered_base_id: str | None,
    ) -> None:
        pulse = int(2 * math.sin(game.time_seconds * 5))
        for base in game.bases.values():
            center = (base.x, base.y)
            radius = self.base_radii[base.id]
            selected = base.id == selected_base_id
            hovered = base.id == hovered_base_id

            if selected:
                pygame.draw.circle(self.screen, MUSTARD, center, radius + 10 + pulse, width=4)
                pygame.draw.circle(self.screen, INK, center, radius + 5, width=2)
            elif hovered:
                pygame.draw.circle(self.screen, MUSTARD, center, radius + 6, width=2)

            pygame.draw.circle(self.screen, INK, center, radius + 3)
            pygame.draw.circle(self.screen, PAPER_LIGHT, center, radius)
            pygame.draw.circle(self.screen, owner_color(base.owner), center, radius - 5)

            text_color = INK if base.owner == NEUTRAL else WHITE
            units = self._render_units(str(int(base.units)), radius, text_color)
            # Sit slightly high so the production badge on the rim never covers it.
            self.screen.blit(units, units.get_rect(center=(base.x, base.y - radius // 8)))

            production_rect = self._draw_production_badge(base.production_rate, base.owner, (base.x, base.y + radius))

            name = self.name_font.render(base.id, True, INK)
            name_rect = name.get_rect(midtop=(base.x, production_rect.bottom + 3)).inflate(
                10 if self.dense_map else 16,
                4,
            )
            pygame.draw.rect(self.screen, PAPER_LIGHT, name_rect, border_radius=3)
            pygame.draw.rect(self.screen, INK, name_rect, width=2, border_radius=3)
            self.screen.blit(name, name.get_rect(center=name_rect.center))

    def _render_units(self, label: str, radius: int, color: tuple[int, int, int]) -> pygame.Surface:
        """Render a unit count as large as the base allows, shrinking long numbers to fit."""

        size = max(13, round(radius * 0.92))
        max_width = 2 * (radius - 7)
        while True:
            if size not in self._count_fonts:
                self._count_fonts[size] = font(size, bold=True)
            text = self._count_fonts[size].render(label, True, color)
            if text.get_width() <= max_width or size <= 11:
                return text
            size -= 1

    def _draw_production_badge(
        self,
        production_rate: float,
        owner: int,
        center: tuple[int, int],
    ) -> pygame.Rect:
        """Draw the production rate as a pill on the base's lower rim."""

        label = f"+{production_rate:g}/s" if production_rate > 0 else "0/s"
        if owner == NEUTRAL:
            fill, text_color = PAPER_LIGHT, INK
        else:
            fill, text_color = owner_color(owner), WHITE
        text = self.production_font.render(label, True, text_color)
        rect = text.get_rect(center=center).inflate(14, 4)
        pygame.draw.rect(self.screen, fill, rect, border_radius=rect.height // 2)
        pygame.draw.rect(self.screen, INK, rect, width=2, border_radius=rect.height // 2)
        self.screen.blit(text, text.get_rect(center=rect.center))
        return rect

    def _draw_fleets(self, game: Game) -> None:
        for fleet in game.fleets:
            x, y = game.fleet_location(fleet)
            source = game.bases[fleet.source_id]
            target = game.bases[fleet.target_id]
            angle = math.atan2(target.y - source.y, target.x - source.x)
            direction = (math.cos(angle), math.sin(angle))
            normal = (-direction[1], direction[0])

            for distance, radius in ((27, 3), (42, 2)):
                trail = (x - direction[0] * distance, y - direction[1] * distance)
                pygame.draw.circle(
                    self.screen,
                    owner_color(fleet.owner),
                    (int(trail[0]), int(trail[1])),
                    radius,
                )

            points = [
                (x + direction[0] * 17, y + direction[1] * 17),
                (x - direction[0] * 12 + normal[0] * 11, y - direction[1] * 12 + normal[1] * 11),
                (x - direction[0] * 8, y - direction[1] * 8),
                (x - direction[0] * 12 - normal[0] * 11, y - direction[1] * 12 - normal[1] * 11),
            ]
            pygame.draw.polygon(self.screen, INK, points)

            inset_points = []
            for point_x, point_y in points:
                inset_points.append((x + (point_x - x) * 0.82, y + (point_y - y) * 0.82))
            pygame.draw.polygon(self.screen, owner_color(fleet.owner), inset_points)

            units = self.small_font.render(str(fleet.units), True, WHITE)
            self.screen.blit(units, units.get_rect(center=(int(x - 2), int(y))))

    def _draw_header(self, game: Game) -> None:
        width = self.screen.get_width()
        pygame.draw.rect(self.screen, PAPER_LIGHT, (0, 0, width, HEADER_HEIGHT))
        pygame.draw.line(self.screen, INK, (0, HEADER_HEIGHT), (width, HEADER_HEIGHT), 2)

        if GREEN in game.players:
            self._draw_three_player_header(game)
            return

        self.screen.blit(self.red_brush, (28, 14))
        self.screen.blit(self.blue_brush, (width - 218, 14))
        self._blit_centered(self._name_label(RED, WHITE, 172), (123, 42))
        self._blit_centered(self._name_label(BLUE, WHITE, 172), (width - 123, 42))

        red_total, red_bases = self._player_summary(game, RED)
        blue_total, blue_bases = self._player_summary(game, BLUE)
        self._draw_player_summary(260, red_total, red_bases, RED_COLOR)
        self._draw_player_summary(width - 260, blue_total, blue_bases, BLUE_COLOR)

        tick = self.small_font.render(f"TICK {game.tick}", True, INK)
        time = self.score_font.render(f"{game.time_seconds:.1f}s", True, INK)
        self._blit_centered(tick, (width // 2, 24))
        self._blit_centered(time, (width // 2, 53))

    def _draw_three_player_header(self, game: Game) -> None:
        width = self.screen.get_width()
        player_columns = (
            (RED, RED_COLOR, int(width * 0.13)),
            (BLUE, BLUE_COLOR, int(width * 0.38)),
            (GREEN, GREEN_COLOR, int(width * 0.63)),
        )
        for player_id, color, center_x in player_columns:
            total_units, base_count = self._player_summary(game, player_id)
            name = self._name_label(player_id, color, int(width * 0.23), font_sizes=(14, 13, 12, 11), two_lines=False)
            units = self.score_font.render(str(total_units), True, color)
            bases = self.small_font.render(
                f"{base_count} base{'s' if base_count != 1 else ''}",
                True,
                MUTED,
            )
            self._blit_centered(name, (center_x, 17))
            self._blit_centered(units, (center_x, 44))
            self._blit_centered(bases, (center_x, 69))

        clock_x = int(width * 0.87)
        tick = self.small_font.render(f"TICK {game.tick}", True, INK)
        time = self.score_font.render(f"{game.time_seconds:.1f}s", True, INK)
        self._blit_centered(tick, (clock_x, 24))
        self._blit_centered(time, (clock_x, 53))

    def _name_label(
        self,
        player_id: int,
        color: tuple[int, int, int],
        max_width: int,
        font_sizes: tuple[int, ...] = (22, 18, 15, 13),
        two_lines: bool = True,
    ) -> pygame.Surface:
        """The player's name in capitals: one line if it fits, else two, else smaller, else cut short."""

        text = self.player_names[player_id].upper()
        for size in font_sizes[:2]:
            label = font(size, bold=True).render(text, True, color)
            if label.get_width() <= max_width:
                return label
        spaces = [index for index, character in enumerate(text) if character == " "]
        if two_lines and spaces:
            # Split at the space nearest the middle.
            split = min(spaces, key=lambda index: abs(index - len(text) / 2))
            lines = (text[:split], text[split + 1 :])
            for size in (18, 15):
                rendered = [font(size, bold=True).render(line, True, color) for line in lines]
                width = max(line.get_width() for line in rendered)
                if width <= max_width:
                    label = pygame.Surface((width, sum(line.get_height() for line in rendered)), pygame.SRCALPHA)
                    label.blit(rendered[0], rendered[0].get_rect(midtop=(width // 2, 0)))
                    label.blit(rendered[1], rendered[1].get_rect(midtop=(width // 2, rendered[0].get_height())))
                    return label
        for size in font_sizes[2:]:
            label = font(size, bold=True).render(text, True, color)
            if label.get_width() <= max_width:
                return label
        while len(text) > 4 and label.get_width() > max_width:
            text = text[:-4] + "..."
            label = font(font_sizes[-1], bold=True).render(text, True, color)
        return label

    def _draw_player_summary(
        self,
        center_x: int,
        total_units: int,
        base_count: int,
        color: tuple[int, int, int],
    ) -> None:
        label = self.small_font.render("TOTAL UNITS", True, MUTED)
        value = self.score_font.render(str(total_units), True, color)
        bases = self.small_font.render(
            f"{base_count} base{'s' if base_count != 1 else ''}",
            True,
            MUTED,
        )
        self._blit_centered(label, (center_x, 18))
        self._blit_centered(value, (center_x, 45))
        self._blit_centered(bases, (center_x, 69))

    def _player_summary(self, game: Game, player_id: int) -> tuple[int, int]:
        units = sum(base.units for base in game.bases.values() if base.owner == player_id)
        units += sum(fleet.units for fleet in game.fleets if fleet.owner == player_id)
        bases = sum(base.owner == player_id for base in game.bases.values())
        return int(units), bases

    def _draw_footer(self) -> None:
        width = self.screen.get_width()
        height = self.screen.get_height()
        top = height - FOOTER_HEIGHT
        pygame.draw.rect(self.screen, PAPER_LIGHT, (0, top, width, FOOTER_HEIGHT))
        pygame.draw.line(self.screen, INK, (0, top), (width, top), 2)

        left = self.body_font.render(
            "LEFT CLICK  Select base · Send half     SHIFT+CLICK  Keep sending all · again to stop",
            True,
            INK,
        )
        right = self.body_font.render("RIGHT CLICK  Cancel     ESC  Quit", True, INK)
        self.screen.blit(left, left.get_rect(midleft=(25, top + FOOTER_HEIGHT // 2)))
        self.screen.blit(right, right.get_rect(midright=(width - 25, top + FOOTER_HEIGHT // 2)))

    def _hovered_base(self, game: Game) -> str | None:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for base in game.bases.values():
            if math.hypot(mouse_x - base.x, mouse_y - base.y) <= self.base_radii[base.id] + 5:
                return base.id
        return None

    def result_buttons(self) -> tuple[pygame.Rect, pygame.Rect]:
        center_x = self.screen.get_width() // 2
        center_y = self.screen.get_height() // 2
        home_button = pygame.Rect(center_x - 190, center_y + 70, 180, 54)
        quit_button = pygame.Rect(center_x + 10, center_y + 70, 180, 54)
        return home_button, quit_button

    def draw_result(self, game: Game, message: str) -> None:
        self._draw_board(game)

        shade = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        shade.fill((42, 40, 37, 105))
        self.screen.blit(shade, (0, 0))

        center_x = self.screen.get_width() // 2
        center_y = self.screen.get_height() // 2
        panel = pygame.Rect(0, 0, 520, 285)
        panel.center = (center_x, center_y)
        pygame.draw.rect(self.screen, PAPER_LIGHT, panel, border_radius=8)
        pygame.draw.rect(self.screen, INK, panel, width=3, border_radius=8)
        pygame.draw.rect(self.screen, PAPER_DARK, panel.inflate(-12, -12), width=2, border_radius=5)

        winner = game.winner()
        if winner == RED:
            brush = self.red_brush
        elif winner == BLUE:
            brush = self.blue_brush
        else:
            brush = None

        if brush is not None:
            brush_rect = brush.get_rect(center=(center_x, center_y - 55))
            self.screen.blit(brush, brush_rect)
            title_color = WHITE
        elif winner == GREEN:
            banner = pygame.Rect(0, 0, 210, 55)
            banner.center = (center_x, center_y - 55)
            pygame.draw.rect(self.screen, GREEN_COLOR, banner, border_radius=5)
            pygame.draw.rect(self.screen, INK, banner, width=2, border_radius=5)
            title_color = WHITE
        else:
            title_color = INK
        title = self.result_font.render(message.upper(), True, title_color)
        self._blit_centered(title, (center_x, center_y - 55))
        # Under the colour, who that was, when the players have names.
        defaults = {RED: "Red", BLUE: "Blue", GREEN: "Green"}
        if winner in defaults and self.player_names[winner] != defaults[winner]:
            name = self.header_font.render(self.player_names[winner], True, INK)
            if name.get_width() > panel.width - 40:
                name = self.small_font.render(self.player_names[winner], True, INK)
            self._blit_centered(name, (center_x, center_y - 4))

        home_button, quit_button = self.result_buttons()
        mouse_position = pygame.mouse.get_pos()
        self._draw_result_button(home_button, "Home screen", home_button.collidepoint(mouse_position))
        self._draw_result_button(quit_button, "Quit", quit_button.collidepoint(mouse_position))
        pygame.display.flip()

    def _draw_result_button(self, rect: pygame.Rect, label: str, hovered: bool) -> None:
        fill = MUSTARD if hovered else INK
        pygame.draw.rect(self.screen, fill, rect, border_radius=6)
        pygame.draw.rect(self.screen, INK, rect, width=2, border_radius=6)
        text = self.button_font.render(label, True, WHITE)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _blit_centered(self, surface: pygame.Surface, center: tuple[int, int]) -> None:
        self.screen.blit(surface, surface.get_rect(center=center))

    def close(self) -> None:
        pygame.quit()
