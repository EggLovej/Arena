"""Splice's own promises beyond what every game makes (test_games.py)."""

import pytest

from arena.live import PERSON, LiveMatch
from arena.registry import load_game

GAME = load_game("splice")  # puts Splice's folder on sys.path, for the imports below

from splice import Layout, cells_in  # noqa: E402


def start(drawing):
    """The opening position of a board drawn as in splice/maps.py; one row numbers its cells 0, 1, 2..."""

    return Layout("test", "Test", "", drawing).start


def owned(position):
    return [set(cells_in(mask)) for mask in position.pieces]


def test_a_grow_adds_a_piece_a_jump_moves_one_and_both_win_every_enemy_neighbour():
    grown = start("1 . 2 . 2 .").play((0, 1))
    assert owned(grown) == [{0, 1, 2}, {4}] and grown.turn == 1
    jumped = start("1 . . 2 . . 2").play((0, 2))
    assert owned(jumped) == [{2, 3}, {6}]
    # Three players: Magenta jumps between Cyan's piece and Amber's and wins both.
    three = start(
        """
         2 . 3 . . 2
        1 . . . . . .
        """
    ).play((6, 1))
    assert owned(three) == [{0, 1, 2}, {5}, set()] and three.turn == 1


def test_a_player_who_cannot_move_misses_their_turn_and_the_last_who_can_takes_the_empty_cells():
    # Cyan is boxed in at the end of the row, so after Magenta comes Amber.
    assert start("2 1 1 . . . 3 . .").play((2, 3)).turn == 2
    # Magenta's grow boxes Cyan in; Magenta alone can move, so the rest of the row is Magenta's.
    ended = start("2 1 . 1 1 . . .").play((1, 2))
    assert owned(ended) == [{1, 2, 3, 4, 5, 6, 7}, {0}] and ended.turn is None


def test_moves_are_checked_and_a_tick_without_one_keeps_the_turn():
    match = GAME.new_match("honeycomb", 0)
    assert match.outcome().ranking == (("magenta", "cyan"),) and match.outcome().reason == "time"
    corner = next(cells_in(match.position.pieces[0]))
    for move, reason in [((1, 2), "cell 1 is not yours"), ((corner, corner), f"cell {corner} is taken"), ((corner, 99), "the board has cells 0 to 57")]:
        assert match.step({"magenta": move}) == {"magenta": reason}
    assert match.step({"magenta": None}) == {}
    assert match.acting() == ("magenta",) and match.tick == 4
    with pytest.raises((TypeError, ValueError)):
        GAME.action_to_json((1, True))


def test_a_person_picks_a_move_that_is_played_on_their_turn():
    live = LiveMatch(GAME, "rift", {"magenta": PERSON, "cyan": "greedy_ai"}, seed=1)
    live.start()
    assert "two cell numbers" in live.command("magenta", [1, 2, 3])
    assert live.step()["tick"] == 1 and live.match.acting() == ("magenta",)  # thinking: the turn stays
    assert live.command("magenta", [0, 1]) is None
    assert live.start()["people"] == {"magenta": {"move": [0, 1]}}
    update = live.step()
    assert update["delta"]["last"]["move"][:4] == ["magenta", 0, 1, 2] and live.match.acting() == ("cyan",)
    live.step()  # Greedy's reply
    assert live.command("magenta", [0, 40]) is None
    assert live.step()["refused"] == {"magenta": "cell 40 is more than two steps from cell 0"}
