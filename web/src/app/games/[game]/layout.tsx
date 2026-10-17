import type { ReactNode } from "react";

import { GameTabs } from "@/components/GameTabs.tsx";
import { getGame, listGames } from "@/lib/data.ts";

export const dynamicParams = false;

export function generateStaticParams() {
  return listGames().map((game) => ({ game: game.id }));
}

export default async function GameLayout({ children, params }: { children: ReactNode; params: Promise<{ game: string }> }) {
  const { game: id } = await params;
  const game = getGame(id);
  return (
    <>
      <div className="game-header">
        <h1>{game.name}</h1>
        <p className="lede">{game.tagline}</p>
      </div>
      <GameTabs gameId={game.id} />
      {children}
    </>
  );
}
