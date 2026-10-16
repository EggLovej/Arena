"""Turn mouse clicks into the same SendAction used by AI files.

Click a base of yours, then a connected base: half of its units go there.
Shift-click the second base instead to start a stream: from then on, every
unit the source has goes to that base, tick after tick, until you
shift-click the source again (or lose it). A stream keeps a front supplied
without clicking again and again. One send per tick is allowed, so a click
always goes first, and several streams take turns.
"""

import math

from game import Game
from game_types import GameView, SendAction
from sizing import base_radii


CLICK_MARGIN = 5
SEND_FRACTION = 0.5


class HumanController:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.selected_base_id: str | None = None
        self._pending_action: SendAction | None = None
        # Running streams, source base -> target base.
        self.streams: dict[str, str] = {}
        self._next_stream = 0  # whose turn it is when several streams can send

    def handle_click(
        self,
        position: tuple[int, int],
        button: int,
        game: Game,
        shift_held: bool = False,
    ) -> None:
        if button == 3:
            self.selected_base_id = None
            return
        if button != 1 or self._pending_action is not None:
            return

        clicked_base_id = self._find_clicked_base(position, game)
        if clicked_base_id is None:
            self.selected_base_id = None
            return

        clicked_base = game.bases[clicked_base_id]
        if self.selected_base_id is None:
            if shift_held and clicked_base_id in self.streams:
                # Shift-click on a streaming base stops its stream.
                del self.streams[clicked_base_id]
                return
            if clicked_base.owner == self.player_id:
                self.selected_base_id = clicked_base_id
            return

        source = game.bases[self.selected_base_id]
        if source.owner != self.player_id:
            self.selected_base_id = None
            return
        if clicked_base_id == source.id:
            if shift_held:
                self.streams.pop(source.id, None)
            self.selected_base_id = None
            return

        if shift_held and game.are_connected(source.id, clicked_base_id):
            # Start (or redirect) this base's stream; it sends from the next tick on.
            self.streams[source.id] = clicked_base_id
            self.selected_base_id = None
        elif game.are_connected(source.id, clicked_base_id) and int(source.units) > 0:
            units = max(1, int(int(source.units) * SEND_FRACTION))
            self._pending_action = SendAction(source.id, clicked_base_id, units)
            self.selected_base_id = None
        elif clicked_base.owner == self.player_id:
            self.selected_base_id = clicked_base_id
        else:
            self.selected_base_id = None

    def choose_action(self, state: GameView) -> SendAction | None:
        if state.player_id != self.player_id:
            raise ValueError("HumanController received the wrong player's view")

        if self.selected_base_id is not None:
            selected = state.get_base(self.selected_base_id)
            if selected.owner != self.player_id:
                self.selected_base_id = None

        action = self._pending_action
        self._pending_action = None
        if action is not None:
            return action
        return self._stream_send(state)

    def _stream_send(self, state: GameView) -> SendAction | None:
        """This tick's send for one of the streams, taking turns; None if none can send."""

        # A base that is no longer ours ends its stream.
        for source_id in [s for s in self.streams if state.get_base(s).owner != self.player_id]:
            del self.streams[source_id]
        streams = list(self.streams.items())
        for offset in range(len(streams)):
            source_id, target_id = streams[(self._next_stream + offset) % len(streams)]
            units = int(state.get_base(source_id).units)
            if units >= 1:
                self._next_stream = (self._next_stream + offset + 1) % len(streams)
                return SendAction(source_id, target_id, units)
        return None

    def _find_clicked_base(self, position: tuple[int, int], game: Game) -> str | None:
        """Return the nearest base whose drawn circle (plus a margin) contains the click."""

        mouse_x, mouse_y = position
        radii = base_radii(game.bases.values())
        hits = []
        for base in game.bases.values():
            distance = math.hypot(mouse_x - base.x, mouse_y - base.y)
            if distance <= radii[base.id] + CLICK_MARGIN:
                hits.append((distance, base.id))
        return min(hits)[1] if hits else None
