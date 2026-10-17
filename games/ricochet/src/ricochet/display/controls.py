"""Gameplay key and mouse bindings, kept next to the hint that describes them."""

import pygame

MOVE_UP, MOVE_LEFT, MOVE_DOWN, MOVE_RIGHT = pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d
SHIELD_KEY = pygame.K_SPACE
FIRE_BUTTON, SHIELD_BUTTON = 0, 2  # Indices into pygame.mouse.get_pressed(3).

TOGGLE_ANALYSIS = pygame.K_F1
TOGGLE_MUTE = pygame.K_m
VOLUME_UP = (pygame.K_EQUALS, pygame.K_PLUS)
VOLUME_DOWN = pygame.K_MINUS

HINT = "WASD move  |  Mouse aim / fire  |  Space shield  |  F1 analysis  |  M mute  |  +/- volume"
