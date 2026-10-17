"""Run AI-vs-AI or human-vs-AI matches."""

import argparse
import sys

from bot_names import bot_name
from game import Game
from game_types import BLUE, GREEN, RED
from loading import available_ais, available_maps, load_ai, load_map


def collect_actions(
    game: Game,
    red_controller,
    blue_controller,
    green_controller=None,
):
    # Every controller receives a snapshot of exactly the same pre-tick state.
    controllers = {
        RED: red_controller,
        BLUE: blue_controller,
        GREEN: green_controller,
    }
    if GREEN in game.players and green_controller is None:
        raise ValueError("A three-player map requires a Green controller")
    return {
        player_id: controllers[player_id](game.view_for(player_id))
        for player_id in game.players
    }


def play_headless(
    game: Game,
    red_controller,
    blue_controller,
    max_ticks: int,
    green_controller=None,
) -> int | None:
    while not game.is_finished() and game.tick < max_ticks:
        actions = collect_actions(
            game,
            red_controller,
            blue_controller,
            green_controller,
        )
        game.step(actions)
    return game.winner()


def play_visual(
    game: Game,
    red_controller,
    blue_controller,
    max_ticks: int,
    speed: float,
    human_controller=None,
    green_controller=None,
    player_names: dict[int, str] | None = None,
) -> tuple[int | None, bool]:
    """``player_names`` labels the players in the window, for example {RED: "Strategic"}."""

    import pygame

    from renderer import Renderer

    renderer = Renderer(game, player_names)
    clock = pygame.time.Clock()
    accumulated_time = 0.0
    running = True

    while running and not game.is_finished() and game.tick < max_ticks:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and human_controller is not None:
                human_controller.handle_click(
                    renderer.map_position(event.pos),
                    event.button,
                    game,
                    shift_held=bool(pygame.key.get_mods() & pygame.KMOD_SHIFT),
                )

        frame_seconds = min(clock.tick(60) / 1000, 0.25)
        accumulated_time += frame_seconds * speed
        while accumulated_time >= game.seconds_per_tick and not game.is_finished():
            actions = collect_actions(
                game,
                red_controller,
                blue_controller,
                green_controller,
            )
            game.step(actions)
            accumulated_time -= game.seconds_per_tick
            if game.tick >= max_ticks:
                break

        selected_base_id = None
        streams = None
        if human_controller is not None:
            selected_base_id = human_controller.selected_base_id
            streams = human_controller.streams
        renderer.draw(game, selected_base_id=selected_base_id, streams=streams)

    winner = game.winner()
    return_home = False
    if running:
        message = player_name(winner) + " wins" if winner is not None else "Draw"
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_h):
                    return_home = True
                    waiting = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    waiting = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    home_button, quit_button = renderer.result_buttons()
                    position = renderer.map_position(event.pos)
                    if home_button.collidepoint(position):
                        return_home = True
                        waiting = False
                    elif quit_button.collidepoint(position):
                        waiting = False
            renderer.draw_result(game, message)
            clock.tick(30)

    renderer.close()
    return winner, return_home


def controller_label(name: str) -> str:
    """How an AI file is shown to people: ``strategic_ai`` -> ``Strategic``."""

    return bot_name(name)


def player_name(player_id: int | None) -> str:
    if player_id == RED:
        return "Red"
    if player_id == BLUE:
        return "Blue"
    if player_id == GREEN:
        return "Green"
    return "No one"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="classic", choices=available_maps())
    parser.add_argument("--red-ai", default="aggressive_ai")
    parser.add_argument("--blue-ai", default="careful_ai")
    parser.add_argument("--green-ai", default="strategic_ai")
    parser.add_argument("--human", choices=("red", "blue", "green"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--tick-seconds", type=float, default=0.1)
    parser.add_argument("--max-seconds", type=float, help="default: the map's own time limit, or 180")
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def configure_from_home_screen(args) -> bool:
    from home_screen import HUMAN, HomeScreen

    selection = HomeScreen(available_maps(), available_ais()).run()
    if selection is None:
        return False

    args.map = selection.map_name
    args.human = None
    selections = {
        "red": selection.red_controller,
        "blue": selection.blue_controller,
        "green": selection.green_controller,
    }
    for color, controller in selections.items():
        if controller is None:
            continue
        if controller == HUMAN:
            args.human = color
        else:
            setattr(args, f"{color}_ai", controller)
    return True


def main() -> None:
    args = parse_args()
    if (args.max_seconds is not None and args.max_seconds <= 0) or args.speed <= 0:
        raise ValueError("max-seconds and speed must be positive")
    if args.human and args.headless:
        raise ValueError("Human play requires the visual window")

    show_home_screen = len(sys.argv) == 1
    while True:
        if show_home_screen and not configure_from_home_screen(args):
            return

        game = Game(load_map(args.map), seconds_per_tick=args.tick_seconds)
        red_controller = load_ai(args.red_ai)
        blue_controller = load_ai(args.blue_ai)
        green_controller = None
        if GREEN in game.players:
            green_controller = load_ai(args.green_ai)
        human_controller = None
        if args.human:
            from human_controller import HumanController

            human_players = {"red": RED, "blue": BLUE, "green": GREEN}
            human_player = human_players[args.human]
            if human_player not in game.players:
                raise ValueError(
                    f"The selected map does not have a {args.human.title()} player"
                )
            human_controller = HumanController(human_player)
            if human_player == RED:
                red_controller = human_controller.choose_action
            elif human_player == BLUE:
                blue_controller = human_controller.choose_action
            else:
                green_controller = human_controller.choose_action
        max_seconds = args.max_seconds or game.map_definition.time_limit or 180
        max_ticks = int(max_seconds / args.tick_seconds)
        names = {RED: controller_label(args.red_ai), BLUE: controller_label(args.blue_ai)}
        if GREEN in game.players:
            names[GREEN] = controller_label(args.green_ai)
        if human_controller is not None:
            names[human_player] = "You"

        if args.headless:
            winner = play_headless(
                game,
                red_controller,
                blue_controller,
                max_ticks,
                green_controller,
            )
            return_home = False
        else:
            winner, return_home = play_visual(
                game,
                red_controller,
                blue_controller,
                max_ticks,
                args.speed,
                human_controller,
                green_controller,
                names,
            )

        if game.is_finished() or game.tick >= max_ticks:
            outcome = f"{names[winner]} ({player_name(winner)}) wins" if winner is not None else "Draw"
        else:
            outcome = "Match closed"
        print(f"{outcome} after {game.tick} ticks ({game.time_seconds:.1f}s)")

        if not return_home:
            return
        show_home_screen = True


if __name__ == "__main__":
    main()
