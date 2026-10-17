"""Portrait avatars for AI players, generated from their names.

Words in the name choose a matching personality (for example "aggressive" gives
horns and gritted teeth). Any other name gets a stable mix of features picked
from a hash of the name, so every new AI file receives its own face.
"""

import hashlib
import math
import re
from dataclasses import dataclass

import pygame

from theme import INK, MUSTARD, PAPER_LIGHT, WHITE, font


CANVAS = 256  # Avatars are drawn at this size and smoothly scaled down.

# Body colour and the glow colour used for eyes and mouth. These are kept away
# from the red, blue, and green player colours so a face never looks like a team.
PALETTE = {
    "rust": ((198, 98, 52), (255, 200, 120)),
    "teal": ((46, 124, 122), (172, 242, 224)),
    "plum": ((112, 70, 126), (246, 214, 120)),
    "slate": ((78, 92, 110), (120, 222, 255)),
    "ochre": ((196, 150, 48), (255, 244, 170)),
    "olive": ((110, 120, 56), (222, 245, 150)),
    "clay": ((164, 104, 86), (255, 214, 190)),
    "sky": ((74, 128, 168), (205, 238, 255)),
}
HEADS = ("round", "box", "hex", "shield")
EYES = ("round", "happy", "angry", "spectacles", "monocle", "visor", "dizzy")
MOUTHS = ("flat", "smile", "smirk", "teeth", "grid", "wave")
TOPS = ("antenna", "horns", "crown", "helmet", "mohawk", "network", "none")

PERSONALITIES = {
    "aggressive": {"color": "rust", "head": "box", "eyes": "angry", "mouth": "teeth", "top": "horns"},
    "careful": {"color": "teal", "head": "round", "eyes": "spectacles", "mouth": "flat", "top": "helmet"},
    "strategic": {"color": "plum", "head": "hex", "eyes": "monocle", "mouth": "smirk", "top": "crown"},
    "neural": {"color": "slate", "head": "box", "eyes": "visor", "mouth": "grid", "top": "network"},
    "random": {"color": "ochre", "head": "round", "eyes": "dizzy", "mouth": "wave", "top": "none"},
}
KEYWORDS = {
    "aggressive": "aggressive", "aggro": "aggressive", "rush": "aggressive",
    "attack": "aggressive", "berserk": "aggressive", "greedy": "aggressive",
    "careful": "careful", "cautious": "careful", "defensive": "careful",
    "defender": "careful", "safe": "careful", "turtle": "careful",
    "strategic": "strategic", "strategy": "strategic", "planner": "strategic",
    "search": "strategic", "mcts": "strategic", "minimax": "strategic", "tactical": "strategic",
    "neural": "neural", "ml": "neural", "nn": "neural", "net": "neural", "learned": "neural",
    "rl": "neural", "ppo": "neural", "dqn": "neural", "deep": "neural",
    "imitation": "neural", "transformer": "neural", "gnn": "neural",
    "random": "random", "rand": "random", "chaos": "random",
}
GENERIC_WORDS = {"ai", "bot", "agent", "player"}


@dataclass(frozen=True)
class AvatarTraits:
    color: str
    head: str
    eyes: str
    mouth: str
    top: str
    seed: int


def avatar_traits(name: str) -> AvatarTraits:
    digest = hashlib.sha256(name.encode()).digest()
    seed = int.from_bytes(digest[:4], "big")

    words = [word for word in re.split(r"[^a-z0-9]+", name.lower()) if word]
    for word in words:
        personality = KEYWORDS.get(word)
        if personality is None:
            continue
        traits = AvatarTraits(seed=seed, **PERSONALITIES[personality])
        if [w for w in words if w not in GENERIC_WORDS] == [personality]:
            return traits
        # A relative of the personality (e.g. "ppo_agent" is neural): keep the
        # colour, eyes, and headgear, but vary the rest so the two can be told apart.
        return AvatarTraits(
            color=traits.color,
            head=HEADS[digest[5] % len(HEADS)],
            eyes=traits.eyes,
            mouth=MOUTHS[digest[7] % len(MOUTHS)],
            top=traits.top,
            seed=seed,
        )

    colors = tuple(PALETTE)
    return AvatarTraits(
        color=colors[digest[4] % len(colors)],
        head=HEADS[digest[5] % len(HEADS)],
        eyes=EYES[digest[6] % len(EYES)],
        mouth=MOUTHS[digest[7] % len(MOUTHS)],
        top=TOPS[digest[8] % len(TOPS)],
        seed=seed,
    )


_cache: dict[tuple[str, int, bool], pygame.Surface] = {}


def avatar_surface(name: str, size: int) -> pygame.Surface:
    """Return the avatar for an AI name at the requested pixel size."""

    return _cached(name, size, human=False)


def human_avatar(size: int) -> pygame.Surface:
    return _cached("human", size, human=True)


def _cached(name: str, size: int, human: bool) -> pygame.Surface:
    key = (name, size, human)
    if key not in _cache:
        canvas = pygame.Surface((CANVAS, CANVAS), pygame.SRCALPHA)
        if human:
            _draw_human(canvas)
        else:
            _draw_robot(canvas, avatar_traits(name))
        _cache[key] = pygame.transform.smoothscale(canvas, (size, size))
    return _cache[key]


# ------------------------------------------------------------------ drawing


def _shade(color, amount):
    """Mix a colour toward black (negative amount) or paper (positive amount)."""

    target = PAPER_LIGHT if amount > 0 else (0, 0, 0)
    amount = abs(amount)
    return tuple(round(c + (t - c) * amount) for c, t in zip(color, target))


def _starburst(canvas, color, center=(128, 150)):
    light, dark = _shade(color, 0.72), _shade(color, 0.6)
    canvas.fill(light)
    wedges = 18
    for index in range(0, wedges, 2):
        start = 2 * math.pi * index / wedges
        end = 2 * math.pi * (index + 1) / wedges
        pygame.draw.polygon(
            canvas,
            dark,
            [
                center,
                (center[0] + 400 * math.cos(start), center[1] + 400 * math.sin(start)),
                (center[0] + 400 * math.cos(end), center[1] + 400 * math.sin(end)),
            ],
        )


def _outlined_polygon(canvas, fill, points, width=8):
    pygame.draw.polygon(canvas, fill, points)
    pygame.draw.polygon(canvas, INK, points, width)


def _draw_robot(canvas, traits: AvatarTraits) -> None:
    body, glow = PALETTE[traits.color]
    _starburst(canvas, body)

    # Shoulders and neck.
    shoulders = pygame.Rect(44, 206, 168, 100)
    pygame.draw.rect(canvas, _shade(body, -0.25), shoulders, border_radius=34)
    pygame.draw.rect(canvas, INK, shoulders, width=8, border_radius=34)
    neck = pygame.Rect(108, 184, 40, 30)
    pygame.draw.rect(canvas, _shade(body, -0.45), neck)
    pygame.draw.rect(canvas, INK, neck, width=6)

    _draw_top(canvas, traits, body, glow, behind_head=True)
    _draw_head(canvas, traits.head, body)
    for x in (54, 202):
        pygame.draw.circle(canvas, _shade(body, -0.35), (x, 132), 13)
        pygame.draw.circle(canvas, INK, (x, 132), 13, 6)
    _draw_top(canvas, traits, body, glow, behind_head=False)

    screen = pygame.Rect(76, 96, 104, 78)
    pygame.draw.rect(canvas, INK, screen, border_radius=20)
    _draw_eyes(canvas, traits, glow)
    _draw_mouth(canvas, traits, glow)


def _draw_head(canvas, head: str, body) -> None:
    rect = pygame.Rect(58, 68, 140, 128)
    if head == "round":
        pygame.draw.ellipse(canvas, body, rect)
        pygame.draw.ellipse(canvas, INK, rect, 8)
    elif head == "box":
        pygame.draw.rect(canvas, body, rect, border_radius=30)
        pygame.draw.rect(canvas, INK, rect, width=8, border_radius=30)
    elif head == "hex":
        _outlined_polygon(canvas, body, [(84, 68), (172, 68), (198, 108), (198, 160), (170, 196), (86, 196), (58, 160), (58, 108)])
    else:  # shield
        _outlined_polygon(canvas, body, [(58, 72), (198, 72), (198, 150), (128, 198), (58, 150)])


def _draw_top(canvas, traits: AvatarTraits, body, glow, behind_head: bool) -> None:
    top = traits.top
    if behind_head:
        if top == "horns":
            bone = (242, 232, 210)
            _outlined_polygon(canvas, bone, [(70, 96), (52, 26), (104, 76)], 6)
            _outlined_polygon(canvas, bone, [(186, 96), (204, 26), (152, 76)], 6)
        elif top == "helmet":
            dome = pygame.Rect(64, 36, 128, 90)
            pygame.draw.ellipse(canvas, _shade(body, -0.3), dome)
            pygame.draw.ellipse(canvas, INK, dome, 8)
        elif top == "antenna":
            pygame.draw.line(canvas, INK, (128, 72), (128, 30), 7)
            pygame.draw.circle(canvas, glow, (128, 26), 13)
            pygame.draw.circle(canvas, INK, (128, 26), 13, 6)
        elif top == "network":
            nodes = [(128, 40), (96, 20), (160, 20)]
            pygame.draw.line(canvas, INK, (128, 72), nodes[0], 7)
            for a, b in ((0, 1), (0, 2), (1, 2)):
                pygame.draw.line(canvas, INK, nodes[a], nodes[b], 5)
            for node in nodes:
                pygame.draw.circle(canvas, glow, node, 10)
                pygame.draw.circle(canvas, INK, node, 10, 5)
        elif top == "mohawk":
            for index in range(4):
                x = 82 + index * 26
                _outlined_polygon(canvas, _shade(body, -0.35), [(x, 80), (x + 13, 34), (x + 26, 80)], 6)
    else:
        if top == "helmet":
            brim = pygame.Rect(50, 68, 156, 22)
            pygame.draw.rect(canvas, _shade(body, -0.45), brim, border_radius=11)
            pygame.draw.rect(canvas, INK, brim, width=6, border_radius=11)
            pygame.draw.circle(canvas, glow, (128, 58), 10)
            pygame.draw.circle(canvas, INK, (128, 58), 10, 5)
        elif top == "crown":
            points = [(82, 82), (86, 34), (106, 60), (128, 24), (150, 60), (170, 34), (174, 82)]
            _outlined_polygon(canvas, MUSTARD, points, 6)
            for x, y in ((86, 34), (128, 24), (170, 34)):
                pygame.draw.circle(canvas, glow, (x, y), 8)
                pygame.draw.circle(canvas, INK, (x, y), 8, 4)
        elif top == "none":
            pass


def _draw_eyes(canvas, traits: AvatarTraits, glow) -> None:
    eyes = traits.eyes
    left, right, y = 106, 150, 122
    if eyes == "round":
        for x in (left, right):
            pygame.draw.circle(canvas, glow, (x, y), 11)
    elif eyes == "happy":
        for x in (left, right):
            pygame.draw.arc(canvas, glow, pygame.Rect(x - 12, y - 8, 24, 22), 0, math.pi, 6)
    elif eyes == "angry":
        for x in (left, right):
            pygame.draw.circle(canvas, glow, (x, y + 2), 12)
        # Slanted brows cut into the tops of both eyes.
        pygame.draw.polygon(canvas, INK, [(90, 104), (122, 104), (122, 124)])
        pygame.draw.polygon(canvas, INK, [(166, 104), (134, 104), (134, 124)])
    elif eyes == "spectacles":
        for x in (left, right):
            pygame.draw.circle(canvas, glow, (x, y), 7)
            pygame.draw.circle(canvas, glow, (x, y), 17, 4)
        pygame.draw.line(canvas, glow, (left + 17, y), (right - 17, y), 4)
    elif eyes == "monocle":
        pygame.draw.line(canvas, glow, (94, y + 2), (118, y - 1), 6)
        pygame.draw.circle(canvas, glow, (right, y), 8)
        pygame.draw.circle(canvas, glow, (right, y), 18, 4)
        pygame.draw.line(canvas, glow, (right + 12, y + 14), (right + 22, y + 42), 3)
    elif eyes == "visor":
        bar = pygame.Rect(88, y - 9, 80, 18)
        pygame.draw.rect(canvas, _shade(glow, -0.55), bar, border_radius=9)
        scan_x = bar.left + 6 + traits.seed % (bar.width - 30)
        pygame.draw.rect(canvas, glow, pygame.Rect(scan_x, bar.top + 2, 18, 14), border_radius=7)
        pygame.draw.rect(canvas, WHITE, pygame.Rect(scan_x + 5, bar.top + 5, 8, 8), border_radius=4)
    else:  # dizzy
        for x in (left, right):
            pygame.draw.line(canvas, glow, (x - 9, y - 9), (x + 9, y + 9), 6)
            pygame.draw.line(canvas, glow, (x - 9, y + 9), (x + 9, y - 9), 6)


def _draw_mouth(canvas, traits: AvatarTraits, glow) -> None:
    mouth = traits.mouth
    y = 154
    if mouth == "flat":
        pygame.draw.line(canvas, glow, (112, y), (144, y), 6)
    elif mouth == "smile":
        pygame.draw.arc(canvas, glow, pygame.Rect(106, y - 20, 44, 28), math.pi * 1.1, math.pi * 1.9, 6)
    elif mouth == "smirk":
        pygame.draw.lines(canvas, glow, False, [(108, y + 2), (136, y + 2), (150, y - 7)], 6)
    elif mouth == "teeth":
        points = [(100 + i * 8, y - 5 if i % 2 else y + 5) for i in range(8)]
        pygame.draw.lines(canvas, glow, False, points, 5)
    elif mouth == "grid":
        for index in range(5):
            cell = pygame.Rect(101 + index * 11, y - 4, 8, 8)
            lit = (traits.seed >> index) & 1 or index == 2
            pygame.draw.rect(canvas, glow if lit else _shade(glow, -0.55), cell, border_radius=2)
    else:  # wave
        points = [(102 + i * 4, y + 5 * math.sin(i * 0.9)) for i in range(14)]
        pygame.draw.lines(canvas, glow, False, points, 5)


def _draw_human(canvas) -> None:
    _starburst(canvas, MUSTARD)
    shoulders = pygame.Rect(40, 170, 176, 130)
    pygame.draw.rect(canvas, INK, shoulders, border_radius=64)
    pygame.draw.circle(canvas, INK, (128, 112), 50)

    label = font(40, bold=True).render("YOU", True, WHITE)
    canvas.blit(label, label.get_rect(center=(128, 222)))
