// The frame around every game page: the list of games on the left, and the
// chosen game's name, tagline and sections beside it.

import Link from "next/link";
import type { ReactNode } from "react";

import { gameView } from "@/games/index.ts";
import { getGame, listGames } from "@/lib/data.ts";

import { GameTabs } from "./GameTabs.tsx";
import { MapPreview } from "./MapPreview.tsx";

export function Explorer({ gameId, home = false, children }: { gameId: string; home?: boolean; children: ReactNode }) {
  const game = getGame(gameId);
  return (
    <div className="explorer">
      <GameList current={gameId} />
      <div className="explorer-main">
        <div className="game-header">
          <h1 className="game-title">{game.name}</h1>
          <p className="game-tagline">{game.tagline}</p>
        </div>
        <GameTabs gameId={gameId} home={home} />
        {children}
      </div>
    </div>
  );
}

function GameList({ current }: { current: string }) {
  return (
    <nav className="game-list" aria-label="Games">
      <h2>Games</h2>
      <ul>
        {listGames().map((summary) => {
          const game = getGame(summary.id);
          const view = gameView(game.id);
          const map = game.maps.find((entry) => entry.id === view.cardMap) ?? game.maps[0];
          return (
            <li key={game.id}>
              <Link href={`/games/${game.id}`} className="game-item" aria-current={game.id === current ? "true" : undefined}>
                <span className="game-item-art">{map && <MapPreview game={game} map={map} />}</span>
                <span className="game-item-text">
                  <strong>{game.name}</strong>
                  <small>{view.genre}</small>
                </span>
              </Link>
            </li>
          );
        })}
        <li className="game-item game-item-soon">More games are on the way.</li>
      </ul>
    </nav>
  );
}
