// The home page: the list of games on the left and a preview of the chosen
// one beside it. Each game's own pages are separate (see GameShell).

import { gameView } from "@/games/index.ts";
import { getGame, listGames } from "@/lib/data.ts";

import { GameExplorer } from "./GameExplorer.tsx";
import { GamePreview } from "./GameOverview.tsx";
import { MapPreview } from "./MapPreview.tsx";

export function Explorer() {
  const games = listGames().map((summary) => {
    const game = getGame(summary.id);
    const view = gameView(game.id);
    const map = game.maps.find((entry) => entry.id === view.cardMap) ?? game.maps[0];
    return { id: game.id, name: game.name, genre: view.genre, art: map && <MapPreview game={game} map={map} /> };
  });
  const previews = Object.fromEntries(games.map((game) => [game.id, <GamePreview key={game.id} gameId={game.id} />]));
  return <GameExplorer games={games} previews={previews} />;
}
