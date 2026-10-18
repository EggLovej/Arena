// A small picture of a map at the start of a match, drawn by the game's own board
// in the theme chosen for the game.

import type { GameData, MapEntry } from "@/lib/types.ts";

import { ThemedBoard } from "./ThemedBoard.tsx";

export function MapPreview({ game, map }: { game: GameData; map: MapEntry }) {
  return (
    <ThemedBoard gameId={game.id} themes={game.themes} board={map.board} snapshot={map.start} tick={0} tickSeconds={game.tick_seconds} seats={game.seats} compact />
  );
}
