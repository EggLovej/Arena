// The frame around a game's own pages: its name, tagline, theme picker and
// sections, in the game's own look when it has one (GameView.look).

import Link from "next/link";
import type { CSSProperties, ReactNode } from "react";

import { cardMap, gameView } from "@/games/index.ts";
import { getGame } from "@/lib/data.ts";

import { GameTabs } from "./GameTabs.tsx";
import { ThemePicker } from "./ThemePicker.tsx";

export function GameShell({ gameId, children }: { gameId: string; children: ReactNode }) {
  const game = getGame(gameId);
  const { look } = gameView(gameId);
  const map = cardMap(game);
  const style: Record<string, string> = {};
  if (look) {
    style.colorScheme = look.scheme;
    for (const [name, value] of Object.entries(look.colors)) style[`--${name}`] = value;
    if (look.font) style["--display"] = look.font.family;
    if (look.backdrop) style.backgroundImage = `url(${look.backdrop})`;
  }
  return (
    <div className={`game-page${look?.font ? ` ${look.font.className}` : ""}`} style={style as CSSProperties}>
      <div className="game-page-inner">
        <Link href="/" className="game-crumb">
          ← All games
        </Link>
        <div className="game-header">
          <div>
            <h1 className="game-title">{game.name}</h1>
            <p className="game-tagline">{game.tagline}</p>
          </div>
          {map && (
            <ThemePicker gameId={gameId} themes={game.themes} map={{ board: map.board, start: map.start }} tickSeconds={game.tick_seconds} seats={game.seats} />
          )}
        </div>
        <GameTabs gameId={gameId} />
        {children}
      </div>
    </div>
  );
}
