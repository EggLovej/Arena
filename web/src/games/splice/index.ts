import type { Snapshot } from "@/lib/types.ts";

import type { GameView } from "../types.ts";
import { SpliceBoard, viewBox, type SpliceBoardData } from "./SpliceBoard.tsx";
import { SpliceControls } from "./SpliceControls.tsx";
import { look } from "./look.ts";

function counts(snapshot: Snapshot): Record<string, number> {
  const result: Record<string, number> = {};
  for (const seat of Object.values((snapshot.pieces ?? {}) as Record<string, string>)) result[seat] = (result[seat] ?? 0) + 1;
  return result;
}

export const splice: GameView = {
  Board: SpliceBoard,
  genre: "Turn-based territory",
  botIcons: ["random_ai", "greedy_ai", "careful_ai", "minimax_ai", "mcts_ai"],
  cardMap: "atoll",
  accent: "#c2307a", // the Petri Dish theme's magenta
  liveMap: "atoll",
  liveSpeed: 1,
  look,
  stats(_board, snapshot) {
    return Object.fromEntries(Object.entries(counts(snapshot)).map(([seat, pieces]) => [seat, [{ label: pieces === 1 ? "piece" : "pieces", value: pieces }]]));
  },
  strength: (_board, snapshot) => counts(snapshot),
  strengthLabel: "Pieces",
  play: {
    Controls: SpliceControls,
    viewBox: (board) => viewBox(board as SpliceBoardData, false),
    help: [
      "On your turn, click a marked cell next to your pieces: a new piece grows there.",
      "Or click one of your pieces, then a cell two steps away: the piece jumps there, leaving its cell empty.",
      "Either way, every enemy piece next to where you land becomes yours. Point at a cell to see what a move would win.",
      "With a keyboard, Tab moves between your pieces and the cells you can move to, and Enter picks one. Escape lets go of a selected piece.",
    ],
    map: "atoll",
  },
};
