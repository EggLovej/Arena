"""A block of progress lines that stays up to date.

In a terminal the block is rewritten in place, so the counters tick over
where they are. Written to a file (a log), it is printed as a whole every
``log_every`` seconds instead, and once more at the end.
"""

import sys
import time


class ProgressBlock:
    def __init__(self, title: str, stream=None, log_every: float = 30.0):
        self.stream = stream or sys.stdout
        self.in_terminal = self.stream.isatty()
        self.log_every = log_every
        self.shown = 0  # lines of the block currently on screen
        self.last_logged = -float("inf")
        print(f"\n=== {title} ===", file=self.stream, flush=True)

    def show(self, lines: list[str], final: bool = False) -> None:
        if self.in_terminal:
            if self.shown:
                # Back to the first line of the block, then write over it.
                self.stream.write(f"\x1b[{self.shown}F")
            for line in lines:
                self.stream.write("\x1b[2K" + line + "\n")
            # A block that got shorter leaves no old lines behind.
            for _ in range(self.shown - len(lines)):
                self.stream.write("\x1b[2K\n")
            self.shown = max(self.shown, len(lines))
        elif final or time.monotonic() - self.last_logged >= self.log_every:
            self.stream.write("\n".join(lines) + "\n\n")
            self.last_logged = time.monotonic()
        self.stream.flush()

    def finish(self, lines: list[str]) -> None:
        """Show the final state; later output starts below it."""

        self.show(lines, final=True)
        self.shown = 0


def duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def count(number: int) -> str:
    """1234567 as 1,234,567."""

    return f"{number:,}"


class LiveLine:
    """One line that shows what is happening right now.

    In a terminal it is rewritten in place (at most ``per_second`` times a
    second, so drawing never slows the work down); in a log it is printed
    every ``log_every`` seconds. ``done`` replaces it with a lasting line.
    """

    def __init__(self, stream=None, per_second: float = 5.0, log_every: float = 30.0):
        self.stream = stream or sys.stdout
        self.in_terminal = self.stream.isatty()
        self.interval = 1.0 / per_second if self.in_terminal else log_every
        self.last = -float("inf")

    def show(self, text: str) -> None:
        now = time.monotonic()
        if now - self.last < self.interval:
            return
        self.last = now
        if self.in_terminal:
            self.stream.write("\r\x1b[2K" + text)
        else:
            self.stream.write(text + "\n")
        self.stream.flush()

    def done(self, text: str = "") -> None:
        """Replace the live line with ``text`` for good; with no text, just clear it."""

        if self.in_terminal:
            self.stream.write("\r\x1b[2K")
        if text:
            self.stream.write(text + "\n")
        self.stream.flush()
        self.last = -float("inf")


def bar(done: int, total: int, width: int = 20) -> str:
    filled = round(width * done / total) if total else width
    return "[" + "#" * filled + "-" * (width - filled) + "]"
