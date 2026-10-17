import Link from "next/link";

import { MapPreview } from "@/components/MapPreview.tsx";
import { gameView } from "@/games/index.ts";
import { getGame, getLeaderboard, listGames } from "@/lib/data.ts";

export default function Home() {
  const games = listGames();
  return (
    <>
      <h1>Games for classic AI bots</h1>
      <p className="lede">
        Every game here is played by bots: scripted strategies, search, and neural networks trained by self-play. Watch
        them play each other and see who leads the ratings.
      </p>
      <div className="game-cards">
        {games.map((summary) => {
          const game = getGame(summary.id);
          const leaderboard = getLeaderboard(summary.id);
          const leader = leaderboard ? Object.values(leaderboard.tables)[0]?.entries[0] : undefined;
          const map = game.maps.find((entry) => entry.id === gameView(game.id).cardMap) ?? game.maps[0];
          return (
            <Link key={game.id} href={`/games/${game.id}`} className="card game-card">
              <div className="game-card-art">{map && <MapPreview game={game} map={map} />}</div>
              <div className="game-card-body">
                <h2>{game.name}</h2>
                <p>{game.tagline}</p>
                <ul className="facts">
                  <li>
                    <strong>{game.bots.length}</strong> bots
                  </li>
                  <li>
                    <strong>{game.maps.length}</strong> maps
                  </li>
                  <li>
                    <strong>{Math.min(...game.maps.map((entry) => entry.seats.length))}</strong>–
                    <strong>{Math.max(...game.maps.map((entry) => entry.seats.length))}</strong> players
                  </li>
                  {leader && (
                    <li>
                      Leader: <strong>{leader.name}</strong>
                    </li>
                  )}
                </ul>
              </div>
            </Link>
          );
        })}
        <div className="card coming-soon">More games are on the way.</div>
      </div>
    </>
  );
}
