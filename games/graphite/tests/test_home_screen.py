import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from avatars import HEADS, MOUTHS, PALETTE, avatar_surface, avatar_traits, human_avatar  # noqa: E402
from home_screen import HUMAN, RANDOM_MAP, START_BUTTON, HomeScreen, MenuSelection, load_ratings  # noqa: E402
from loading import available_ais, available_maps  # noqa: E402


def click(position):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=position, button=1)


class AvatarTests(unittest.TestCase):
    def test_personality_words_choose_matching_features(self):
        self.assertEqual(avatar_traits("aggressive_ai").top, "horns")
        self.assertEqual(avatar_traits("careful_ai").eyes, "spectacles")
        self.assertEqual(avatar_traits("strategic_ai").top, "crown")
        self.assertEqual(avatar_traits("neural_ai").eyes, "visor")

    def test_avatars_are_stable_for_a_name(self):
        self.assertEqual(avatar_traits("my_new_ai"), avatar_traits("my_new_ai"))

    def test_related_names_keep_the_family_look_but_differ(self):
        canonical = avatar_traits("neural_ai")
        relative = avatar_traits("ppo_neural_agent")
        self.assertEqual((relative.color, relative.eyes, relative.top), (canonical.color, canonical.eyes, canonical.top))
        self.assertIn(relative.head, HEADS)
        self.assertIn(relative.mouth, MOUTHS)

    def test_unknown_names_still_get_valid_avatars(self):
        traits = avatar_traits("zephyr")
        self.assertIn(traits.color, PALETTE)

    def test_avatar_surfaces_have_the_requested_size(self):
        pygame.init()
        pygame.display.set_mode((10, 10))
        for name in available_ais():
            self.assertEqual(avatar_surface(name, 90).get_size(), (90, 90))
        self.assertEqual(human_avatar(64).get_size(), (64, 64))


class RatingTests(unittest.TestCase):
    def test_ratings_come_from_the_file_and_are_empty_without_a_good_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.json"
            self.assertEqual(load_ratings(path), {})
            path.write_text(json.dumps({"ratings": {"strategic_ai": 1000.4, "beast_ai": 1473}}))
            self.assertEqual(load_ratings(path), {"strategic_ai": 1000, "beast_ai": 1473})
            path.write_text("not json")
            self.assertEqual(load_ratings(path), {})

    def test_the_player_page_draws_with_and_without_ratings(self):
        for ratings in ({}, {name: 1000 for name in available_ais()}):
            home = HomeScreen(available_maps(), available_ais(), ratings=ratings)
            home.page = "players"
            home._draw()
            pygame.quit()


class HomeScreenTests(unittest.TestCase):
    def setUp(self):
        self.home = HomeScreen(available_maps(), available_ais())

    def tearDown(self):
        pygame.quit()

    def choose_map(self, name):
        tiles = dict(self.home.map_tiles())
        self.home.handle_event(click(tiles[name].center))

    def pick(self, controller):
        tiles = dict(self.home.roster_tiles())
        self.home.handle_event(click(tiles[controller].center))

    def test_every_map_has_a_tile_plus_random(self):
        values = [value for value, _ in self.home.map_tiles()]
        self.assertEqual(values, [*available_maps(), RANDOM_MAP])

    def test_clicking_a_map_opens_player_select(self):
        self.choose_map("siege")
        self.assertEqual(self.home.page, "players")
        self.assertEqual(self.home.map_name, "siege")
        self.assertEqual(self.home.active_colors(), ("red", "blue"))

    def test_random_tile_picks_an_installed_map(self):
        self.choose_map(RANDOM_MAP)
        self.assertIn(self.home.map_name, available_maps())

    def test_picking_players_returns_a_three_player_selection(self):
        self.choose_map("triangle")
        self.pick("strategic_ai")  # Red, then the Blue card becomes active
        self.pick(HUMAN)  # Blue
        self.pick("careful_ai")  # Green

        selection = self.home.handle_event(click(START_BUTTON.center))

        self.assertEqual(
            selection,
            MenuSelection("triangle", "strategic_ai", HUMAN, "careful_ai"),
        )

    def test_only_one_colour_can_be_human(self):
        self.choose_map("triangle")
        self.pick(HUMAN)  # Red
        self.pick(HUMAN)  # Blue: taken, so nothing changes
        self.assertNotEqual(self.home.controllers["blue"], HUMAN)
        self.assertEqual(self.home.active_slot, "blue")

    def test_two_player_maps_leave_green_empty(self):
        self.choose_map("classic")
        selection = self.home.handle_event(click(START_BUTTON.center))
        self.assertIsNone(selection.green_controller)

    def test_escape_on_player_select_goes_back_to_maps(self):
        self.choose_map("ring")
        self.home.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        self.assertEqual(self.home.page, "maps")

    def test_arrow_keys_and_enter_choose_a_map(self):
        self.home.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        self.home.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
        self.assertEqual(self.home.map_name, available_maps()[1])


if __name__ == "__main__":
    unittest.main()
