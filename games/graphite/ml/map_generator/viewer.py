"""Browse generated maps, drawn exactly as the game draws them.

Run from the game directory:

    .venv/bin/python -m ml maps
    .venv/bin/python -m ml maps --seed 42 --asymmetry 0.3
"""

import argparse
import random

import pygame

from game import Game
from game_types import BLUE as BLUE_PLAYER, RED as RED_PLAYER
from renderer import FOOTER_HEIGHT, HEADER_HEIGHT, Renderer
from theme import BLUE, INK, MUSTARD, MUTED, PAPER_LIGHT, RED, WHITE

from ml.map_generator.generator import (
    DEFAULT_PARAMS,
    GeneratedMap,
    MapGenParams,
    generate_map,
    save_map,
)


ASYMMETRY_STEP = 0.1
MESSAGE_SECONDS = 3.0


class MapView(Renderer):
    """The in-game board, with the header and footer showing map details instead."""

    def __init__(self, generated: GeneratedMap, asymmetry_is_fixed: bool):
        self.generated = generated
        self.asymmetry_is_fixed = asymmetry_is_fixed
        self.game = Game(generated.definition)
        self.message = ""
        super().__init__(self.game)
        pygame.display.set_caption("Map generator")

        top = self.screen.get_height() - FOOTER_HEIGHT + 8
        right = self.screen.get_width()
        self.previous_button = pygame.Rect(20, top, 130, FOOTER_HEIGHT - 16)
        self.save_button = pygame.Rect(right - 310, top, 140, FOOTER_HEIGHT - 16)
        self.next_button = pygame.Rect(right - 150, top, 130, FOOTER_HEIGHT - 16)

    def draw_map(self) -> None:
        self.draw(self.game)

    def _draw_header(self, game: Game) -> None:
        width = self.screen.get_width()
        pygame.draw.rect(self.screen, PAPER_LIGHT, (0, 0, width, HEADER_HEIGHT))
        pygame.draw.line(self.screen, INK, (0, HEADER_HEIGHT), (width, HEADER_HEIGHT), 2)

        self.screen.blit(self.red_brush, (28, 14))
        self.screen.blit(self.blue_brush, (width - 218, 14))
        self._blit_centered(self.header_font.render("RED", True, WHITE), (123, 42))
        self._blit_centered(self.header_font.render("BLUE", True, WHITE), (width - 123, 42))
        self._draw_player_summary(260, *self._player_summary(game, RED_PLAYER), RED)
        self._draw_player_summary(width - 260, *self._player_summary(game, BLUE_PLAYER), BLUE)

        definition = self.generated.definition
        if self.generated.asymmetry == 0:
            symmetry = "mirrored"
        else:
            symmetry = f"asymmetry {self.generated.asymmetry:.2f}"
        if self.asymmetry_is_fixed:
            symmetry += " (fixed)"

        seed = self.header_font.render(f"SEED {self.generated.seed}", True, INK)
        size = self.small_font.render(
            f"{len(definition.bases)} bases · {len(definition.edges)} routes",
            True,
            INK,
        )
        self._blit_centered(seed, (width // 2, 22))
        self._blit_centered(size, (width // 2, 47))
        self._blit_centered(self.small_font.render(symmetry, True, MUTED), (width // 2, 67))

    def _draw_footer(self) -> None:
        width = self.screen.get_width()
        top = self.screen.get_height() - FOOTER_HEIGHT
        pygame.draw.rect(self.screen, PAPER_LIGHT, (0, top, width, FOOTER_HEIGHT))
        pygame.draw.line(self.screen, INK, (0, top), (width, top), 2)

        self._draw_result_button(self.previous_button, "Previous", self._hovered(self.previous_button))
        self._draw_result_button(self.save_button, "Save map", self._hovered(self.save_button))
        self._draw_result_button(self.next_button, "Next", self._hovered(self.next_button))

        hint = self.message or "LEFT / RIGHT  Browse     UP / DOWN  Asymmetry     S  Save     ESC  Quit"
        color = INK if self.message else MUTED
        text = self.small_font.render(hint, True, color)
        centre_x = (self.previous_button.right + self.save_button.left) // 2
        self._blit_centered(text, (centre_x, top + FOOTER_HEIGHT // 2))

    def _draw_result_button(self, rect: pygame.Rect, label: str, hovered: bool) -> None:
        # Same look as the game's buttons, with a smaller label to fit the footer.
        pygame.draw.rect(self.screen, MUSTARD if hovered else INK, rect, border_radius=6)
        pygame.draw.rect(self.screen, INK, rect, width=2, border_radius=6)
        text = self.body_font.render(label, True, WHITE)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _hovered(self, rect: pygame.Rect) -> bool:
        return rect.collidepoint(pygame.mouse.get_pos())


class MapViewer:
    """Steps through seeds one at a time; every seed always shows the same map."""

    def __init__(
        self,
        seed: int,
        asymmetry: float | None = None,
        params: MapGenParams = DEFAULT_PARAMS,
    ):
        self.seed = seed
        self.asymmetry = asymmetry
        self.params = params
        self.view = self._build_view()

    def run(self) -> None:
        clock = pygame.time.Clock()
        message_time_left = 0.0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key in (pygame.K_RIGHT, pygame.K_SPACE, pygame.K_RETURN):
                        self.show_seed(self.seed + 1)
                    elif event.key == pygame.K_LEFT:
                        self.show_seed(self.seed - 1)
                    elif event.key == pygame.K_UP:
                        self.change_asymmetry(+1)
                    elif event.key == pygame.K_DOWN:
                        self.change_asymmetry(-1)
                    elif event.key == pygame.K_s:
                        message_time_left = self.save()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.view.next_button.collidepoint(event.pos):
                        self.show_seed(self.seed + 1)
                    elif self.view.previous_button.collidepoint(event.pos):
                        self.show_seed(self.seed - 1)
                    elif self.view.save_button.collidepoint(event.pos):
                        message_time_left = self.save()

            if message_time_left > 0:
                message_time_left -= clock.get_time() / 1000
                if message_time_left <= 0:
                    self.view.message = ""

            self.view.draw_map()
            clock.tick(30)

        self.view.close()

    def show_seed(self, seed: int) -> None:
        self.seed = max(0, seed)
        self.view = self._build_view()

    def change_asymmetry(self, direction: int) -> None:
        """Step through: usual mix, then fixed levels 0.0, 0.1, ... 1.0."""

        if self.asymmetry is None:
            if direction < 0:
                return
            self.asymmetry = 0.0
        else:
            level = round(self.asymmetry + direction * ASYMMETRY_STEP, 2)
            self.asymmetry = None if level < 0 else min(level, 1.0)
        self.view = self._build_view()

    def save(self) -> float:
        path = save_map(self.view.generated.definition)
        self.view.message = f"Saved to maps/{path.name}"
        return MESSAGE_SECONDS

    def _build_view(self) -> MapView:
        generated = generate_map(self.seed, self.params, self.asymmetry)
        return MapView(generated, asymmetry_is_fixed=self.asymmetry is not None)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, help="First seed to show (default: random)")
    parser.add_argument(
        "--asymmetry",
        type=float,
        help="Fix the asymmetry level, 0 to 1 (default: the generator's usual mix)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None and args.seed < 0:
        raise ValueError("seed cannot be negative")
    if args.asymmetry is not None and not 0 <= args.asymmetry <= 1:
        raise ValueError("asymmetry must be between 0 and 1")

    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    MapViewer(seed, args.asymmetry).run()


if __name__ == "__main__":
    main()
