// A small picture of a map at the start of a match, drawn by the game's own board.

import { gameView } from "@/games/index.ts";
import type { GameData, MapEntry } from "@/lib/types.ts";

export function MapPreview({ game, map }: { game: GameData; map: MapEntry }) {
  const { Board } = gameView(game.id);
  return <Board board={map.board} snapshot={map.start} tick={0} tickSeconds={game.tick_seconds} theme={game.theme} seats={game.seats} compact />;
}
