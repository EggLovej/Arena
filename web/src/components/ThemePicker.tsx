"use client";

// The choice of theme on a game's pages, each theme shown as a small board drawn
// in it. The choice holds for every board of the game: previews, replays, play.

import { gameView } from "@/games/index.ts";
import { chooseTheme } from "@/lib/themes.ts";
import type { MapEntry, Seat, Theme } from "@/lib/types.ts";

import { useTheme } from "./ThemedBoard.tsx";

type Props = {
  gameId: string;
  themes: Theme[];
  map: Pick<MapEntry, "board" | "start">; // the map each theme is shown on
  tickSeconds: number;
  seats: Seat[];
};

export function ThemePicker({ gameId, themes, map, tickSeconds, seats }: Props) {
  const { Board } = gameView(gameId);
  const [current, known] = useTheme(gameId, themes);
  if (themes.length < 2) return null;
  return (
    <fieldset className="theme-picker" data-pending-theme={known ? undefined : gameId}>
      <legend>Theme</legend>
      {themes.map((theme) => (
        <label key={theme.id} className="theme-option" title={theme.description}>
          <input
            type="radio"
            name={`${gameId}-theme`}
            checked={theme.id === current.id}
            onChange={() => chooseTheme(gameId, themes, theme.id)}
          />
          <span className="theme-option-art">
            <Board board={map.board} snapshot={map.start} tick={0} tickSeconds={tickSeconds} theme={theme} seats={seats} compact />
          </span>
          {theme.name}
        </label>
      ))}
    </fieldset>
  );
}
