"""Atlas-style Pygame home screen: pick a map, then pick who plays each colour.

Step 1 works like a fighting game's stage select: hovering a tile shows a large
preview of that map. Step 2 works like a character select: click a colour card,
then click the human or AI portrait that should play it. Each AI shows its
estimated rating (ELO, strategic AI = 1000) when ``ais/ratings.json`` has one;
``python -m ml bot-ratings`` writes that file.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pygame

from avatars import avatar_surface, human_avatar
from bot_names import bot_name
from game_types import GREEN as GREEN_PLAYER, NEUTRAL, RED as RED_PLAYER, MapDefinition
from loading import load_map
from sizing import base_radii
from theme import (
    BLUE,
    DISABLED,
    GREEN,
    INK,
    MUSTARD,
    MUTED,
    PAPER_DARK,
    PAPER_LIGHT,
    RED,
    WHITE,
    font,
    load_background,
    load_brush,
    owner_color,
)


HUMAN = "human"
RATINGS_FILE = Path(__file__).resolve().parent / "ais" / "ratings.json"
RANDOM_MAP = "__random__"
WINDOW_SIZE = (1000, 800)
HEADER_HEIGHT = 96

# Step 1: map select
PREVIEW_RECT = pygame.Rect(30, 116, 600, 356)
INFO_RECT = pygame.Rect(650, 116, 320, 356)
GRID_RECT = pygame.Rect(30, 494, 940, 262)
PLATE_HEIGHT = 24

# Step 2: player select
BACK_BUTTON = pygame.Rect(30, 112, 150, 46)
MAP_CHIP = pygame.Rect(196, 112, 420, 46)
ROSTER_RECT = pygame.Rect(30, 176, 940, 250)
SLOTS_RECT = pygame.Rect(30, 448, 940, 254)
START_BUTTON = pygame.Rect(720, 722, 250, 56)

COLORS = ("red", "blue", "green")
COLOR_STYLE = {"red": ("RED", RED), "blue": ("BLUE", BLUE), "green": ("GREEN", GREEN)}
HOVER_FILL = (244, 235, 214)

_QUIT = object()


@dataclass(frozen=True)
class MenuSelection:
    map_name: str
    red_controller: str
    blue_controller: str
    green_controller: str | None


def load_ratings(path: Path = RATINGS_FILE) -> dict[str, int]:
    """Each AI's estimated rating by name; empty when there is no readable ratings file."""

    try:
        return {name: round(value) for name, value in json.loads(path.read_text())["ratings"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


class HomeScreen:
    def __init__(
        self,
        map_names: tuple[str, ...],
        ai_names: tuple[str, ...],
        ratings: dict[str, int] | None = None,
    ):
        if not map_names:
            raise ValueError("The home screen needs at least one map")
        if not ai_names:
            raise ValueError("The home screen needs at least one AI")

        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Graphite")
        self.clock = pygame.time.Clock()
        self.background = load_background(WINDOW_SIZE)
        self.red_brush = load_brush("red-brush-banner.png", (175, 50))
        self.blue_brush = load_brush("blue-brush-banner.png", (175, 50))

        self.title_font = font(38, bold=True)
        self.kicker_font = font(14, bold=True)
        self.label_font = font(20, bold=True)
        self.map_name_font = font(32, bold=True)
        self.stat_label_font = font(12, bold=True)
        self.stat_value_font = font(24, bold=True)
        self.plate_font = font(15, bold=True)
        self.tile_font = font(13, bold=True)
        self.small_font = font(15)
        self.button_font = font(21, bold=True)
        self.question_font = font(240, bold=True)

        self.map_names = map_names
        self.map_definitions = {name: load_map(name) for name in map_names}
        self.controller_names = (HUMAN, *ai_names)
        self.ratings = load_ratings() if ratings is None else ratings
        self.controllers = {"red": HUMAN, "blue": ai_names[0], "green": ai_names[0]}

        self.page = "maps"
        self.map_name = map_names[0]
        self.focus = 0
        self.active_slot = "red"
        self._images: dict[tuple[str, tuple[int, int], bool], pygame.Surface] = {}
        self._rng = random.Random()

    # ------------------------------------------------------------------ loop

    def run(self) -> MenuSelection | None:
        while True:
            for event in pygame.event.get():
                result = self.handle_event(event)
                if result is _QUIT:
                    pygame.quit()
                    return None
                if isinstance(result, MenuSelection):
                    return result

            self._draw()
            self.clock.tick(60)

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.QUIT:
            return _QUIT
        if self.page == "maps":
            return self._handle_map_event(event)
        return self._handle_player_event(event)

    # ------------------------------------------------------------ step 1: maps

    def map_tiles(self) -> list[tuple[str, pygame.Rect]]:
        values = [*self.map_names, RANDOM_MAP]
        columns = self._grid_columns()
        rows = math.ceil(len(values) / columns)
        gap = 12
        width = (GRID_RECT.width - gap * (columns - 1)) // columns
        height = min(125, (GRID_RECT.height - gap * (rows - 1)) // rows)
        return [
            (
                value,
                pygame.Rect(
                    GRID_RECT.left + (index % columns) * (width + gap),
                    GRID_RECT.top + (index // columns) * (height + gap),
                    width,
                    height,
                ),
            )
            for index, value in enumerate(values)
        ]

    def _grid_columns(self) -> int:
        count = len(self.map_names) + 1
        return 6 if count <= 12 else 8 if count <= 24 else 10

    def _handle_map_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            index = self._tile_index_at(event.pos)
            if index is not None:
                self.focus = index
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            index = self._tile_index_at(event.pos)
            if index is not None:
                self._choose_map(index)
        elif event.type == pygame.KEYDOWN:
            columns = self._grid_columns()
            last = len(self.map_names)
            if event.key == pygame.K_ESCAPE:
                return _QUIT
            if event.key == pygame.K_LEFT:
                self.focus = max(0, self.focus - 1)
            elif event.key == pygame.K_RIGHT:
                self.focus = min(last, self.focus + 1)
            elif event.key == pygame.K_UP and self.focus - columns >= 0:
                self.focus -= columns
            elif event.key == pygame.K_DOWN:
                self.focus = min(last, self.focus + columns)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._choose_map(self.focus)
        return None

    def _tile_index_at(self, position) -> int | None:
        for index, (_, rect) in enumerate(self.map_tiles()):
            if rect.collidepoint(position):
                return index
        return None

    def _choose_map(self, index: int) -> None:
        value = self.map_tiles()[index][0]
        self.map_name = self._rng.choice(self.map_names) if value == RANDOM_MAP else value
        self.focus = self.map_names.index(self.map_name)
        self.page = "players"
        self.active_slot = self.active_colors()[0]
        self._ensure_single_human()

    # ---------------------------------------------------------- step 2: players

    def active_colors(self) -> tuple[str, ...]:
        return COLORS if self._map_has_green() else COLORS[:2]

    def roster_tiles(self) -> list[tuple[str, pygame.Rect]]:
        count = len(self.controller_names)
        columns = min(count, 6)
        rows = math.ceil(count / columns)
        gap = 16
        width = min(160, (ROSTER_RECT.width - gap * (columns - 1)) // columns)
        # Tall enough for a square portrait plus the name plate, and no taller.
        height = min(width + PLATE_HEIGHT + 10, (ROSTER_RECT.height - 14 * (rows - 1)) // rows)
        left = ROSTER_RECT.centerx - (columns * width + gap * (columns - 1)) // 2
        top = ROSTER_RECT.centery - (rows * height + 14 * (rows - 1)) // 2
        return [
            (
                value,
                pygame.Rect(
                    left + (index % columns) * (width + gap),
                    top + (index // columns) * (height + 14),
                    width,
                    height,
                ),
            )
            for index, value in enumerate(self.controller_names)
        ]

    def slot_cards(self) -> list[tuple[str, pygame.Rect]]:
        colors = self.active_colors()
        gap = 24
        width = min(300, (SLOTS_RECT.width - gap * (len(colors) - 1)) // len(colors))
        left = SLOTS_RECT.centerx - (len(colors) * width + gap * (len(colors) - 1)) // 2
        return [
            (color, pygame.Rect(left + index * (width + gap), SLOTS_RECT.top, width, SLOTS_RECT.height))
            for index, color in enumerate(colors)
        ]

    def _handle_player_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                self.page = "maps"
            elif event.key == pygame.K_RETURN:
                return self._selection()
            elif event.key == pygame.K_TAB:
                colors = self.active_colors()
                self.active_slot = colors[(colors.index(self.active_slot) + 1) % len(colors)]
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if BACK_BUTTON.collidepoint(event.pos):
                self.page = "maps"
            elif START_BUTTON.collidepoint(event.pos):
                return self._selection()
            for color, rect in self.slot_cards():
                if rect.collidepoint(event.pos):
                    self.active_slot = color
            for value, rect in self.roster_tiles():
                if rect.collidepoint(event.pos):
                    self._assign(value)
        return None

    def _assign(self, controller: str) -> None:
        if not self._option_enabled(self.active_slot, controller):
            return
        self.controllers[self.active_slot] = controller
        colors = self.active_colors()
        position = colors.index(self.active_slot)
        if position + 1 < len(colors):
            self.active_slot = colors[position + 1]

    def _option_enabled(self, color: str, controller: str) -> bool:
        if controller != HUMAN:
            return True
        return all(
            self.controllers[other] != HUMAN
            for other in self.active_colors()
            if other != color
        )

    def _ensure_single_human(self) -> None:
        seen_human = False
        for color in self.active_colors():
            if self.controllers[color] == HUMAN:
                if seen_human:
                    self.controllers[color] = self.controller_names[1]
                seen_human = True

    def _selection(self) -> MenuSelection:
        return MenuSelection(
            map_name=self.map_name,
            red_controller=self.controllers["red"],
            blue_controller=self.controllers["blue"],
            green_controller=self.controllers["green"] if self._map_has_green() else None,
        )

    def _map_has_green(self, map_name: str | None = None) -> bool:
        definition = self.map_definitions[map_name or self.map_name]
        return any(base.owner == GREEN_PLAYER for base in definition.bases)

    # ------------------------------------------------------------------ drawing

    def _draw(self) -> None:
        self.screen.blit(self.background, (0, 0))
        if self.page == "maps":
            self._draw_header("STEP 1 OF 2", "CHOOSE A MAP")
            self._draw_map_page()
        else:
            self._draw_header("STEP 2 OF 2", "CHOOSE PLAYERS")
            self._draw_player_page()
        pygame.display.flip()

    def _draw_header(self, kicker: str, title: str) -> None:
        width = WINDOW_SIZE[0]
        pygame.draw.rect(self.screen, PAPER_LIGHT, (0, 0, width, HEADER_HEIGHT))
        pygame.draw.line(self.screen, INK, (0, HEADER_HEIGHT - 1), (width, HEADER_HEIGHT - 1), 2)
        self.screen.blit(self.red_brush, (30, 24))
        self.screen.blit(self.blue_brush, (width - 205, 24))
        self._blit_centered(self.label_font.render("RED", True, WHITE), (117, 49))
        self._blit_centered(self.label_font.render("BLUE", True, WHITE), (width - 117, 49))
        self._blit_centered(self.kicker_font.render(f"AGENT MAP GAME  ·  {kicker}", True, MUTED), (width // 2, 26))
        self._blit_centered(self.title_font.render(title, True, INK), (width // 2, 62))

    def _draw_panel(self, rect: pygame.Rect, fill=PAPER_LIGHT) -> None:
        pygame.draw.rect(self.screen, INK, rect.inflate(6, 6), border_radius=8)
        pygame.draw.rect(self.screen, fill, rect, border_radius=6)
        pygame.draw.rect(self.screen, PAPER_DARK, rect.inflate(-10, -10), width=2, border_radius=4)

    # step 1 -------------------------------------------------------------------

    def _draw_map_page(self) -> None:
        tiles = self.map_tiles()
        focused = tiles[self.focus][0]
        self._draw_preview(focused)
        self._draw_map_info(focused)

        for index, (value, rect) in enumerate(tiles):
            if index != self.focus:
                self._draw_map_tile(value, rect, focused=False)
        # The focused tile pops out above its neighbours.
        value, rect = tiles[self.focus]
        self._draw_map_tile(value, rect, focused=True)

        hint = self.small_font.render(
            "Hover to preview  ·  Click or Enter to choose  ·  Arrow keys move  ·  Esc quits",
            True,
            MUTED,
        )
        self._blit_centered(hint, (WINDOW_SIZE[0] // 2, 777))

    def _draw_preview(self, value: str) -> None:
        self._draw_panel(PREVIEW_RECT)
        inner = PREVIEW_RECT.inflate(-28, -28)
        if value == RANDOM_MAP:
            self._blit_centered(self.question_font.render("?", True, MUSTARD), inner.center)
        else:
            self.screen.blit(self._map_image(value, inner.size, detailed=True), inner.topleft)

        name = "Random map" if value == RANDOM_MAP else _display_name(value)
        plate = self.map_name_font.render(name.upper(), True, WHITE)
        plate_rect = plate.get_rect(bottomleft=(PREVIEW_RECT.left + 24, PREVIEW_RECT.bottom + 8)).inflate(28, 12)
        pygame.draw.rect(self.screen, INK, plate_rect, border_radius=6)
        pygame.draw.rect(self.screen, MUSTARD, plate_rect, width=3, border_radius=6)
        self._blit_centered(plate, plate_rect.center)

    def _draw_map_info(self, value: str) -> None:
        self._draw_panel(INFO_RECT)
        left = INFO_RECT.left + 26
        self.screen.blit(self.kicker_font.render("MAP DETAILS", True, MUTED), (left, INFO_RECT.top + 22))

        if value == RANDOM_MAP:
            lines = (
                f"Picks one of the {len(self.map_names)}",
                "installed maps at random.",
                "",
                "You will see which one",
                "before the match starts.",
            )
            for index, line in enumerate(lines):
                text = self.small_font.render(line, True, INK)
                self.screen.blit(text, (left, INFO_RECT.top + 62 + index * 24))
            return

        definition = self.map_definitions[value]
        stats = _map_stats(definition)
        players = [color for color in COLORS if color != "green" or self._map_has_green(value)]
        x = left
        for color in players:
            pygame.draw.circle(self.screen, INK, (x + 11, INFO_RECT.top + 66), 11)
            pygame.draw.circle(self.screen, COLOR_STYLE[color][1], (x + 11, INFO_RECT.top + 66), 8)
            x += 28
        label = self.label_font.render(f"{len(players)} PLAYERS", True, INK)
        self.screen.blit(label, label.get_rect(midleft=(x + 6, INFO_RECT.top + 66)))

        cells = (
            ("BASES", str(stats["bases"])),
            ("ROUTES", str(stats["routes"])),
            ("NEUTRAL", str(stats["neutral"])),
            ("RICHEST", f"+{stats['richest']:g}/s"),
            ("AVG ROUTE", f"{stats['average_route']:.1f}s"),
            ("START ARMY", f"{stats['start_army']:g}"),
        )
        for index, (name, number) in enumerate(cells):
            column, row = index % 2, index // 2
            cell_x = left + column * 140
            cell_y = INFO_RECT.top + 106 + row * 70
            self.screen.blit(self.stat_label_font.render(name, True, MUTED), (cell_x, cell_y))
            self.screen.blit(self.stat_value_font.render(number, True, INK), (cell_x, cell_y + 18))

        hint = self.small_font.render("Bigger dots produce more units", True, MUTED)
        self.screen.blit(hint, (left, INFO_RECT.bottom - 40))

    def _draw_map_tile(self, value: str, rect: pygame.Rect, focused: bool) -> None:
        if focused:
            rect = rect.inflate(10, 10)
            shadow = rect.move(5, 6)
            pygame.draw.rect(self.screen, (120, 108, 90), shadow, border_radius=8)

        border = MUSTARD if focused else INK
        pygame.draw.rect(self.screen, border, rect, border_radius=8)
        inner = rect.inflate(-8 if focused else -6, -8 if focused else -6)
        fill = HOVER_FILL if focused else PAPER_LIGHT
        pygame.draw.rect(self.screen, fill, inner, border_radius=5)

        picture = pygame.Rect(inner.left + 4, inner.top + 4, inner.width - 8, inner.height - PLATE_HEIGHT - 6)
        if value == RANDOM_MAP:
            question = font(max(24, picture.height - 10), bold=True).render("?", True, MUSTARD)
            self._blit_centered(question, picture.center)
        else:
            self.screen.blit(self._map_image(value, picture.size, detailed=False), picture.topleft)
            if self._map_has_green(value):
                badge = self.tile_font.render("3P", True, INK)
                badge_rect = badge.get_rect(topright=(inner.right - 6, inner.top + 6)).inflate(10, 4)
                pygame.draw.rect(self.screen, MUSTARD, badge_rect, border_radius=badge_rect.height // 2)
                pygame.draw.rect(self.screen, INK, badge_rect, width=2, border_radius=badge_rect.height // 2)
                self._blit_centered(badge, badge_rect.center)

        plate = pygame.Rect(inner.left, inner.bottom - PLATE_HEIGHT, inner.width, PLATE_HEIGHT)
        pygame.draw.rect(self.screen, INK, plate, border_bottom_left_radius=5, border_bottom_right_radius=5)
        name = "RANDOM" if value == RANDOM_MAP else _display_name(value).upper()
        self._blit_centered(self.tile_font.render(name, True, MUSTARD if focused else WHITE), plate.center)

    def _map_image(self, map_name: str, size: tuple[int, int], detailed: bool) -> pygame.Surface:
        key = (map_name, size, detailed)
        if key not in self._images:
            self._images[key] = render_map_image(self.map_definitions[map_name], size, detailed)
        return self._images[key]

    # step 2 -------------------------------------------------------------------

    def _draw_player_page(self) -> None:
        mouse = pygame.mouse.get_pos()
        self._draw_back_button(BACK_BUTTON.collidepoint(mouse))
        self._draw_map_chip()

        hint = self.small_font.render("Pick a colour card, then pick who plays it", True, MUTED)
        self.screen.blit(hint, hint.get_rect(midright=(970, MAP_CHIP.centery)))

        for value, rect in self.roster_tiles():
            self._draw_roster_tile(value, rect, rect.collidepoint(mouse))
        for color, rect in self.slot_cards():
            self._draw_slot_card(color, rect)

        note = self.small_font.render(
            "One human seat  ·  Tab switches card  ·  Esc goes back  ·  Enter starts",
            True,
            MUTED,
        )
        self.screen.blit(note, note.get_rect(midleft=(30, START_BUTTON.centery)))

        fill = MUSTARD if START_BUTTON.collidepoint(mouse) else INK
        pygame.draw.rect(self.screen, fill, START_BUTTON, border_radius=6)
        pygame.draw.rect(self.screen, INK, START_BUTTON, width=2, border_radius=6)
        self._blit_centered(self.button_font.render("START MATCH", True, WHITE), START_BUTTON.center)

    def _draw_back_button(self, hovered: bool) -> None:
        pygame.draw.rect(self.screen, MUSTARD if hovered else INK, BACK_BUTTON, border_radius=6)
        arrow_x, arrow_y = BACK_BUTTON.left + 26, BACK_BUTTON.centery
        pygame.draw.polygon(self.screen, WHITE, [(arrow_x - 8, arrow_y), (arrow_x + 6, arrow_y - 9), (arrow_x + 6, arrow_y + 9)])
        label = self.button_font.render("MAPS", True, WHITE)
        self.screen.blit(label, label.get_rect(midleft=(arrow_x + 20, arrow_y)))

    def _draw_map_chip(self) -> None:
        self._draw_panel(MAP_CHIP)
        thumbnail = pygame.Rect(MAP_CHIP.left + 8, MAP_CHIP.top + 5, 62, MAP_CHIP.height - 10)
        self.screen.blit(self._map_image(self.map_name, thumbnail.size, detailed=False), thumbnail.topleft)
        name = self.label_font.render(_display_name(self.map_name).upper(), True, INK)
        self.screen.blit(name, name.get_rect(midleft=(thumbnail.right + 14, MAP_CHIP.centery)))
        stats = _map_stats(self.map_definitions[self.map_name])
        detail = self.small_font.render(
            f"{len(self.active_colors())} players  ·  {stats['bases']} bases",
            True,
            MUTED,
        )
        self.screen.blit(detail, detail.get_rect(midleft=(thumbnail.right + 24 + name.get_width(), MAP_CHIP.centery)))

    def _draw_roster_tile(self, controller: str, rect: pygame.Rect, hovered: bool) -> None:
        enabled = self._option_enabled(self.active_slot, controller)
        picked_by = [color for color in self.active_colors() if self.controllers[color] == controller]
        highlighted = hovered and enabled

        border = MUSTARD if highlighted else INK
        pygame.draw.rect(self.screen, border, rect, border_radius=8)
        inner = rect.inflate(-6, -6)
        pygame.draw.rect(self.screen, HOVER_FILL if highlighted else PAPER_LIGHT, inner, border_radius=6)

        size = min(inner.width - 12, inner.height - PLATE_HEIGHT - 16)
        portrait = human_avatar(size) if controller == HUMAN else avatar_surface(controller, size)
        portrait_rect = portrait.get_rect(midtop=(inner.centerx, inner.top + 6))
        self.screen.blit(portrait, portrait_rect)
        pygame.draw.rect(self.screen, INK, portrait_rect, width=2)

        plate = pygame.Rect(inner.left, inner.bottom - PLATE_HEIGHT - 2, inner.width, PLATE_HEIGHT + 2)
        pygame.draw.rect(self.screen, INK, plate, border_bottom_left_radius=6, border_bottom_right_radius=6)
        label = self.tile_font.render(_controller_name(controller).upper(), True, MUSTARD if highlighted else WHITE)
        self._blit_centered(label, plate.center)

        if controller in self.ratings:
            self._draw_rating_badge(self.ratings[controller], topright=(rect.right - 7, rect.top + 7))

        if not enabled:
            shade = pygame.Surface(rect.size, pygame.SRCALPHA)
            shade.fill((249, 243, 230, 170))
            self.screen.blit(shade, rect.topleft)
            taken = self.plate_font.render("TAKEN", True, DISABLED)
            self._blit_centered(taken, portrait_rect.center)

        # Player tokens, like coins dropped on a fighter's portrait.
        for index, color in enumerate(picked_by):
            center = (rect.left + 18 + index * 26, rect.top + 18)
            pygame.draw.circle(self.screen, INK, center, 14)
            pygame.draw.circle(self.screen, COLOR_STYLE[color][1], center, 11)
            self._blit_centered(self.tile_font.render(COLOR_STYLE[color][0][0], True, WHITE), center)

    def _draw_slot_card(self, color: str, rect: pygame.Rect) -> None:
        label, player_color = COLOR_STYLE[color]
        active = color == self.active_slot

        if active:
            pulse = 2 + int(2 * (1 + math.sin(pygame.time.get_ticks() / 180)))
            pygame.draw.rect(self.screen, MUSTARD, rect.inflate(10 + pulse * 2, 10 + pulse * 2), border_radius=12)
        self._draw_panel(rect)

        band = pygame.Rect(rect.left, rect.top, rect.width, 46)
        pygame.draw.rect(self.screen, player_color, band, border_top_left_radius=6, border_top_right_radius=6)
        pygame.draw.line(self.screen, INK, band.bottomleft, band.bottomright, 3)
        self.screen.blit(self.label_font.render(label, True, WHITE), (band.left + 18, band.centery - 12))
        if active:
            tag = self.tile_font.render("PICKING", True, INK)
            tag_rect = tag.get_rect(midright=(band.right - 14, band.centery)).inflate(16, 8)
            pygame.draw.rect(self.screen, MUSTARD, tag_rect, border_radius=tag_rect.height // 2)
            pygame.draw.rect(self.screen, INK, tag_rect, width=2, border_radius=tag_rect.height // 2)
            self._blit_centered(tag, tag_rect.center)

        controller = self.controllers[color]
        name = self.plate_font.render(_controller_name(controller).upper(), True, WHITE)
        plate_height = name.get_height() + 10
        rating = self.ratings.get(controller)
        rating_height = 30 if rating is not None else 0
        size = min(rect.width - 80, rect.height - band.height - plate_height - rating_height - 50)
        portrait = human_avatar(size) if controller == HUMAN else avatar_surface(controller, size)
        # Centre the portrait, its name plate, and the rating together in the card body.
        content_top = band.bottom + (rect.bottom - band.bottom - (size + 14 + plate_height + rating_height)) // 2
        portrait_rect = portrait.get_rect(midtop=(rect.centerx, content_top))
        pygame.draw.rect(self.screen, INK, portrait_rect.inflate(6, 6), border_radius=4)
        self.screen.blit(portrait, portrait_rect)

        plate = name.get_rect(midtop=(rect.centerx, portrait_rect.bottom + 14 + 5)).inflate(26, 10)
        pygame.draw.rect(self.screen, INK, plate, border_radius=plate.height // 2)
        self._blit_centered(name, plate.center)
        if rating is not None:
            text = self.small_font.render(f"Estimated ELO {rating}", True, MUTED)
            self.screen.blit(text, text.get_rect(midtop=(rect.centerx, plate.bottom + 10)))

    def _draw_rating_badge(self, rating: int, topright: tuple[int, int]) -> None:
        """A small "ELO 1234" pill, like a sticker on the portrait's corner."""

        label = self.stat_label_font.render(f"ELO {rating}", True, INK)
        badge = label.get_rect().inflate(12, 6)
        badge.topright = topright
        pygame.draw.rect(self.screen, MUSTARD, badge, border_radius=badge.height // 2)
        pygame.draw.rect(self.screen, INK, badge, width=2, border_radius=badge.height // 2)
        self._blit_centered(label, badge.center)

    def _blit_centered(self, surface: pygame.Surface, center) -> None:
        self.screen.blit(surface, surface.get_rect(center=center))


def render_map_image(
    map_definition: MapDefinition,
    size: tuple[int, int],
    detailed: bool,
    supersample: int = 2,
) -> pygame.Surface:
    """Draw a map's routes and bases scaled into `size`, with bases sized by production."""

    width, height = size[0] * supersample, size[1] * supersample
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    bases = map_definition.bases
    radii = base_radii(bases)
    margin = max(radii.values()) + 6
    left = min(base.x for base in bases) - margin
    top = min(base.y for base in bases) - margin
    span_x = max(base.x for base in bases) + margin - left
    span_y = max(base.y for base in bases) + margin - top
    scale = min(width / span_x, height / span_y)
    offset_x = (width - span_x * scale) / 2
    offset_y = (height - span_y * scale) / 2

    def point(base):
        return (offset_x + (base.x - left) * scale, offset_y + (base.y - top) * scale)

    positions = {base.id: point(base) for base in bases}
    outer = max(2, round(7 * scale))
    for edge in map_definition.edges:
        pygame.draw.line(surface, INK, positions[edge.a], positions[edge.b], outer)
        if detailed and outer >= 5:
            pygame.draw.line(surface, PAPER_DARK, positions[edge.a], positions[edge.b], outer - 4)

    ring = max(2, round(3 * scale))
    for base in bases:
        center = positions[base.id]
        radius = max(2 * supersample, radii[base.id] * scale)
        pygame.draw.circle(surface, INK, center, radius + ring)
        if detailed:
            pygame.draw.circle(surface, PAPER_LIGHT, center, radius)
            pygame.draw.circle(surface, owner_color(base.owner), center, max(1, radius - ring * 1.5))
        else:
            pygame.draw.circle(surface, owner_color(base.owner), center, radius)

    return pygame.transform.smoothscale(surface, size)


def _map_stats(definition: MapDefinition) -> dict[str, float]:
    owners = [base.owner for base in definition.bases]
    times = [edge.travel_time for edge in definition.edges]
    return {
        "bases": len(definition.bases),
        "routes": len(definition.edges),
        "neutral": owners.count(NEUTRAL),
        "richest": max(base.production_rate for base in definition.bases),
        "average_route": sum(times) / len(times),
        # Every map has Red; symmetric maps give everyone the same army.
        "start_army": sum(base.units for base in definition.bases if base.owner == RED_PLAYER),
    }


def _display_name(name: str) -> str:
    return name.replace("_", " ").title()


def _controller_name(name: str) -> str:
    if name == HUMAN:
        return "Human player"
    return bot_name(name)
