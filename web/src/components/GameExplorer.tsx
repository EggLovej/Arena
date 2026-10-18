"use client";

// Choosing a game in the list shows its preview; choosing the shown game again
// opens its page.

import Link from "next/link";
import { useState, type ReactNode } from "react";

import { Chevron } from "./Chevron.tsx";

type Entry = { id: string; name: string; genre: string; art: ReactNode };

export function GameExplorer({ games, previews }: { games: Entry[]; previews: Record<string, ReactNode> }) {
  const [shown, setShown] = useState(games[0]?.id);
  return (
    <div className="explorer">
      <nav className="game-list" aria-label="Games">
        <h2>Games</h2>
        <ul>
          {games.map((game) => (
            <li key={game.id}>
              <Link
                href={`/games/${game.id}`}
                className="game-item"
                aria-current={game.id === shown ? "true" : undefined}
                title={game.id === shown ? `Open the ${game.name} page` : `Show ${game.name}`}
                onClick={(event) => {
                  if (game.id === shown) return;
                  event.preventDefault();
                  setShown(game.id);
                }}
              >
                <span className="game-item-art">{game.art}</span>
                <span className="game-item-text">
                  <strong>{game.name}</strong>
                  <small>{game.genre}</small>
                </span>
                <Chevron className="game-item-open" />
              </Link>
            </li>
          ))}
          <li className="game-item game-item-soon">More games are on the way.</li>
        </ul>
      </nav>
      <div className="explorer-main">{shown && previews[shown]}</div>
    </div>
  );
}
