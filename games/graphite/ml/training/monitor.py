"""Live charts for a training run.

Run from the game directory, while training is running or afterwards:

    .venv/bin/python -m ml monitor --run my_first_run

The window re-reads the run's metrics file whenever it changes, so it can be
opened and closed at any time without affecting training. It draws one chart
per group of metrics; a metric named ``score/vs careful_ai`` is the line
"vs careful_ai" in the chart "score".

Move the mouse over a chart to read exact values. T switches between charts
and tables. Escape closes the window.
"""

import argparse
import math

import pygame

from ml.training.runlog import METRICS_FILE, RUNS_DIRECTORY, format_duration, read_metrics
from theme import INK, MUTED, PAPER, PAPER_DARK, PAPER_LIGHT, font


WINDOW_SIZE = (1200, 820)
HEADER_HEIGHT = 64
FOOTER_HEIGHT = 34
MAX_CHARTS_IN_TWO_COLUMNS = 6  # more than this and the charts go three to a row
REFRESH_SECONDS = 1.0

# Line colours, always assigned in this order and never reused within a chart.
# Checked for colour-blind separation on the paper background.
SERIES_COLORS = ((42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0))
MAX_SERIES = len(SERIES_COLORS)
GRID = PAPER_DARK

# Groups whose values are shares between 0 and 1; they get a fixed 0-100% axis.
SHARE_GROUPS = ("score", "wins", "agreement", "decided games", "training wins", "promotion match", "behaviour", "comparison score")


class Monitor:
    def __init__(self, run: str):
        self.run = run
        self.folder = RUNS_DIRECTORY / run
        self.rounds: list[dict] = []
        self.loaded_version = None
        self.show_tables = False

        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(f"Training monitor: {run}")
        self.title_font = font(24, bold=True)
        self.panel_font = font(18, bold=True)
        self.text_font = font(15)
        self.value_font = font(15, bold=True)
        self.tick_font = font(13)

    def run_forever(self) -> None:
        clock = pygame.time.Clock()
        since_refresh = REFRESH_SECONDS
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_t:
                        self.show_tables = not self.show_tables

            since_refresh += clock.get_time() / 1000
            if since_refresh >= REFRESH_SECONDS:
                self.refresh()
                since_refresh = 0.0
            self.draw(pygame.mouse.get_pos())
            pygame.display.flip()
            clock.tick(30)
        pygame.quit()

    def refresh(self) -> None:
        """Re-read the metrics file, but only if it has changed."""

        path = self.folder / METRICS_FILE
        version = path.stat().st_mtime if path.exists() else None
        if version != self.loaded_version:
            self.rounds = read_metrics(self.folder) if version is not None else []
            self.loaded_version = version

    def draw(self, mouse: tuple[int, int]) -> None:
        self.screen.fill(PAPER)
        self._draw_header()
        self._draw_footer()

        groups = group_metrics(self.rounds)
        if not groups:
            waiting = "Waiting for the first round to finish" if self.loaded_version else f"Waiting for run '{self.run}' to start"
            text = self.panel_font.render(waiting, True, MUTED)
            self.screen.blit(text, text.get_rect(center=self.screen.get_rect().center))
            return

        columns = 2 if len(groups) <= MAX_CHARTS_IN_TWO_COLUMNS else 3
        rows = math.ceil(len(groups) / columns)
        gap = 14
        width = (WINDOW_SIZE[0] - gap * (columns + 1)) // columns
        height = (WINDOW_SIZE[1] - HEADER_HEIGHT - FOOTER_HEIGHT - gap * (rows + 1)) // rows
        hovered = None
        for index, (group, series) in enumerate(groups.items()):
            rect = pygame.Rect(
                gap + (index % columns) * (width + gap),
                HEADER_HEIGHT + gap + (index // columns) * (height + gap),
                width,
                height,
            )
            if self.show_tables:
                self._draw_table(rect, group, series)
            else:
                readout = self._draw_chart(rect, group, series, mouse)
                hovered = hovered or readout
        if hovered:
            self._draw_tooltip(*hovered)

    def _draw_header(self) -> None:
        width = WINDOW_SIZE[0]
        pygame.draw.rect(self.screen, PAPER_LIGHT, (0, 0, width, HEADER_HEIGHT))
        pygame.draw.line(self.screen, INK, (0, HEADER_HEIGHT), (width, HEADER_HEIGHT), 2)
        title = self.title_font.render(self.run, True, INK)
        self.screen.blit(title, title.get_rect(midleft=(20, HEADER_HEIGHT // 2)))
        if self.rounds:
            latest = self.rounds[-1]
            status = f"round {latest['round']}  ·  {format_duration(latest['seconds'])} of training"
            text = self.text_font.render(status, True, MUTED)
            self.screen.blit(text, text.get_rect(midright=(width - 20, HEADER_HEIGHT // 2)))

    def _draw_footer(self) -> None:
        width, height = WINDOW_SIZE
        hint = "T  charts / tables     ESC  close     hover a chart for exact values"
        text = self.text_font.render(hint, True, MUTED)
        self.screen.blit(text, text.get_rect(midleft=(20, height - FOOTER_HEIGHT // 2)))

    def _draw_card(self, rect: pygame.Rect, group: str, series: dict[str, list]) -> int:
        """The frame, title, and legend shared by charts and tables. Returns the y below them."""

        pygame.draw.rect(self.screen, PAPER_LIGHT, rect, border_radius=6)
        pygame.draw.rect(self.screen, GRID, rect, width=1, border_radius=6)
        title = self.panel_font.render(group.capitalize(), True, INK)
        self.screen.blit(title, (rect.left + 14, rect.top + 10))
        if len(series) > MAX_SERIES:
            note = self.text_font.render(f"showing {MAX_SERIES} of {len(series)} lines", True, MUTED)
            self.screen.blit(note, note.get_rect(topright=(rect.right - 14, rect.top + 12)))

        # Legend: a short line in the series colour, the name, then the newest value.
        x = rect.left + 14
        y = rect.top + 38
        for color, (name, points) in zip(SERIES_COLORS, series.items()):
            label = self.text_font.render(name, True, MUTED)
            value = self.value_font.render(format_value(points[-1][1], group), True, INK)
            needed = 22 + label.get_width() + 6 + value.get_width() + 18
            if x + needed > rect.right - 8:
                x = rect.left + 14
                y += 20
            pygame.draw.line(self.screen, color, (x, y + 9), (x + 16, y + 9), 3)
            self.screen.blit(label, (x + 22, y))
            self.screen.blit(value, (x + 22 + label.get_width() + 6, y))
            x += needed
        return y + 26

    def _draw_chart(self, rect: pygame.Rect, group: str, series: dict[str, list], mouse):
        top = self._draw_card(rect, group, series)
        plot = pygame.Rect(rect.left + 56, top + 6, rect.width - 56 - 62, rect.bottom - top - 6 - 40)
        shown = dict(list(series.items())[:MAX_SERIES])

        values = [value for points in shown.values() for _, value in points]
        low, high, ticks = value_axis(values, group)
        first, last = self.rounds[0]["round"], self.rounds[-1]["round"]
        span = max(1, last - first)

        def x_of(round_number: float) -> float:
            return plot.left + (round_number - first) / span * plot.width

        def y_of(value: float) -> float:
            return plot.bottom - (value - low) / (high - low) * plot.height

        for tick in ticks:
            y = round(y_of(tick))
            pygame.draw.line(self.screen, GRID, (plot.left, y), (plot.right, y), 1)
            label = self.tick_font.render(format_value(tick, group), True, MUTED)
            self.screen.blit(label, label.get_rect(midright=(plot.left - 8, y)))
        for tick in round_ticks(first, last):
            label = self.tick_font.render(str(tick), True, MUTED)
            self.screen.blit(label, label.get_rect(midtop=(x_of(tick), plot.bottom + 6)))
        axis_name = self.tick_font.render("round", True, MUTED)
        self.screen.blit(axis_name, axis_name.get_rect(topright=(plot.right, plot.bottom + 6 + 15)))

        ends = []
        for color, (name, points) in zip(SERIES_COLORS, shown.items()):
            path = [(x_of(round_number), y_of(value)) for round_number, value in points]
            if len(path) > 1:
                pygame.draw.lines(self.screen, color, False, path, 2)
            # The newest point gets a dot with a ring in the background colour,
            # so it stays readable where lines cross.
            end = (round(path[-1][0]), round(path[-1][1]))
            pygame.draw.circle(self.screen, PAPER_LIGHT, end, 6)
            pygame.draw.circle(self.screen, color, end, 4)
            ends.append((end, points[-1][1]))

        # Newest values are written beside the line ends, unless they would overlap.
        heights = sorted(end[1] for end, _ in ends)
        if all(b - a >= 16 for a, b in zip(heights, heights[1:])):
            for end, value in ends:
                label = self.value_font.render(format_value(value, group), True, INK)
                self.screen.blit(label, label.get_rect(midleft=(end[0] + 9, end[1])))

        if not plot.inflate(20, 20).collidepoint(mouse):
            return None
        # The crosshair snaps to the nearest round this chart has a value for,
        # so there is no need to aim at a line.
        nearest = min(
            {round_number for points in shown.values() for round_number, _ in points},
            key=lambda round_number: abs(x_of(round_number) - mouse[0]),
        )
        x = round(x_of(nearest))
        pygame.draw.line(self.screen, MUTED, (x, plot.top), (x, plot.bottom), 1)
        rows = []
        for color, (name, points) in zip(SERIES_COLORS, shown.items()):
            at_round = dict(points).get(nearest)
            if at_round is not None:
                pygame.draw.circle(self.screen, PAPER_LIGHT, (x, round(y_of(at_round))), 6)
                pygame.draw.circle(self.screen, color, (x, round(y_of(at_round))), 4)
                rows.append((color, name, format_value(at_round, group)))
        return (x, mouse[1], f"round {nearest}", rows) if rows else None

    def _draw_tooltip(self, x: int, y: int, heading: str, rows: list) -> None:
        heading_text = self.text_font.render(heading, True, MUTED)
        lines = [
            (color, self.value_font.render(value, True, INK), self.text_font.render(name, True, MUTED))
            for color, name, value in rows
        ]
        width = max([heading_text.get_width()] + [24 + v.get_width() + 8 + n.get_width() for _, v, n in lines]) + 24
        height = 14 + 20 + 20 * len(lines) + 8
        box = pygame.Rect(x + 14, y - height // 2, width, height)
        if box.right > WINDOW_SIZE[0] - 8:
            box.right = x - 14
        box.clamp_ip(pygame.Rect(8, HEADER_HEIGHT + 8, WINDOW_SIZE[0] - 16, WINDOW_SIZE[1] - HEADER_HEIGHT - 16))

        pygame.draw.rect(self.screen, PAPER_LIGHT, box, border_radius=5)
        pygame.draw.rect(self.screen, INK, box, width=1, border_radius=5)
        self.screen.blit(heading_text, (box.left + 12, box.top + 8))
        for index, (color, value, name) in enumerate(lines):
            line_y = box.top + 32 + index * 20
            pygame.draw.line(self.screen, color, (box.left + 12, line_y + 9), (box.left + 26, line_y + 9), 3)
            self.screen.blit(value, (box.left + 34, line_y))
            self.screen.blit(name, (box.left + 34 + value.get_width() + 8, line_y))

    def _draw_table(self, rect: pygame.Rect, group: str, series: dict[str, list]) -> None:
        top = self._draw_card(rect, group, series)
        names = list(series)
        by_round = {name: dict(points) for name, points in series.items()}
        row_height = 19
        visible = max(1, (rect.bottom - top - 28) // row_height)
        rounds = [entry["round"] for entry in self.rounds][-visible:]

        column = (rect.width - 28 - 60) // max(1, len(names))
        header_y = top + 2
        self.screen.blit(self.tick_font.render("round", True, MUTED), (rect.left + 14, header_y))
        for index, name in enumerate(names):
            label = self.tick_font.render(name, True, MUTED)
            self.screen.blit(label, label.get_rect(topright=(rect.left + 14 + 60 + (index + 1) * column, header_y)))
        pygame.draw.line(self.screen, GRID, (rect.left + 14, header_y + 17), (rect.right - 14, header_y + 17), 1)

        for line, round_number in enumerate(rounds):
            y = header_y + 21 + line * row_height
            self.screen.blit(self.text_font.render(str(round_number), True, MUTED), (rect.left + 14, y))
            for index, name in enumerate(names):
                value = by_round[name].get(round_number)
                text = format_value(value, group) if value is not None else "-"
                label = self.text_font.render(text, True, INK)
                self.screen.blit(label, label.get_rect(topright=(rect.left + 14 + 60 + (index + 1) * column, y)))


def group_metrics(rounds: list[dict]) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Arrange logged rounds as group -> line name -> [(round, value), ...]."""

    groups: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for entry in rounds:
        for key, value in entry.items():
            if "/" not in key or not isinstance(value, (int, float)):
                continue
            group, name = key.split("/", 1)
            groups.setdefault(group, {}).setdefault(name, []).append((entry["round"], value))

    # Some groups are only logged every few rounds. The round that logged the
    # most decides the order, so such a group keeps the place its trainer gave it.
    fullest = max(rounds, key=len, default={})
    order = list(dict.fromkeys(key.split("/", 1)[0] for key in fullest if "/" in key))
    return dict(sorted(groups.items(), key=lambda item: order.index(item[0]) if item[0] in order else len(order)))


def value_axis(values: list[float], group: str) -> tuple[float, float, list[float]]:
    """Lowest and highest value on the axis, and where to draw grid lines."""

    if group in SHARE_GROUPS:
        return 0.0, 1.0, [0.0, 0.25, 0.5, 0.75, 1.0]
    low, high = min(values), max(values)
    if high - low < 1e-9:
        low, high = low - 0.5, high + 0.5
    # Step sizes people read easily: 1, 2, or 5 times a power of ten.
    rough = (high - low) / 4
    magnitude = 10 ** math.floor(math.log10(rough))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= rough)
    low = math.floor(low / step) * step
    high = math.ceil(high / step) * step
    count = round((high - low) / step)
    return low, high, [low + index * step for index in range(count + 1)]


def round_ticks(first: int, last: int) -> list[int]:
    span = last - first
    if span <= 0:
        return [first]
    step = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000) if span / s <= 8)
    start = math.ceil(first / step) * step
    return list(range(start, last + 1, step))


def format_value(value: float, group: str) -> str:
    if group in SHARE_GROUPS:
        return f"{value:.0%}"
    return f"{value:.3g}" if abs(value) < 100 else f"{value:,.0f}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="Name of the run in ml/runs/")
    parser.add_argument("--wait", action="store_true", help="Open even if the run has not started yet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.wait and not (RUNS_DIRECTORY / args.run / METRICS_FILE).exists():
        runs = sorted(path.parent.name for path in RUNS_DIRECTORY.glob(f"*/{METRICS_FILE}"))
        raise SystemExit(f"No run named '{args.run}'. Runs found: {', '.join(runs) or 'none'}")
    Monitor(args.run).run_forever()


if __name__ == "__main__":
    main()
