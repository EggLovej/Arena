// The frame around a game's own pages: its name, tagline and sections, in the
// game's own look when it has one (GameView.look).

import Link from "next/link";
import type { CSSProperties, ReactNode } from "react";

import { gameView } from "@/games/index.ts";
import { getGame } from "@/lib/data.ts";

import { GameTabs } from "./GameTabs.tsx";

export function GameShell({ gameId, children }: { gameId: string; children: ReactNode }) {
  const game = getGame(gameId);
  const { look, accent } = gameView(gameId);
  const style: Record<string, string> = {};
  if (accent) style["--stage-accent"] = accent;
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
          <h1 className="game-title">{game.name}</h1>
          <p className="game-tagline">{game.tagline}</p>
        </div>
        <GameTabs gameId={gameId} />
        {children}
      </div>
    </div>
  );
}
