// A game's front page, shared by every game: a featured match playing on a
// loop, the top of the ladder, the ways to play or watch, and more matches.

import Link from "next/link";
import type { CSSProperties } from "react";

import { gameView } from "@/games/index.ts";
import { botNames, getGame, getLeaderboard, getMatches } from "@/lib/data.ts";
import { featuredMatches } from "@/lib/featured.ts";

import { Ladder } from "./Ladder.tsx";
import { LiveMatch } from "./LiveMatch.tsx";
import { MapPreview } from "./MapPreview.tsx";
import { MatchCard } from "./MatchCard.tsx";

const LIVE_SPEED = 2;

export function GameOverview({ gameId }: { gameId: string }) {
  const game = getGame(gameId);
  const view = gameView(gameId);
  const leaderboard = getLeaderboard(gameId);
  const featured = featuredMatches(getMatches(gameId), leaderboard);
  const names = botNames(game);
  const [live, ...more] = featured;
  const liveMap = live && game.maps.find((entry) => entry.id === live.map);
  const cardMap = game.maps.find((entry) => entry.id === view.cardMap) ?? game.maps[0];
  const tables = Object.values(leaderboard?.tables ?? {}).map((table) => ({
    players: table.players,
    label: table.label,
    entries: table.entries.slice(0, 5).map(({ bot, name, kind, rating }) => ({ bot, name, kind, rating })),
  }));
  const note = leaderboard ? `${leaderboard.matches} matches, updated ${new Date(leaderboard.updated).toUTCString().slice(5, 16)}` : "";

  return (
    <>
      <div className="overview" style={view.accent ? ({ "--stage-accent": view.accent } as CSSProperties) : undefined}>
        <section className="overview-stage" aria-label="Featured match">
          {live && liveMap ? (
            <LiveMatch
              gameId={game.id}
              gameName={game.name}
              replayUrl={`/data/${game.id}/replays/${live.id}.json`}
              map={liveMap}
              lineup={live.lineup}
              seats={game.seats}
              botNames={names}
              theme={game.theme}
              tickSeconds={game.tick_seconds}
              speed={LIVE_SPEED}
            />
          ) : (
            <div className="stage">{cardMap && <MapPreview game={game} map={cardMap} />}</div>
          )}
          <p className="stage-legend">
            {live && liveMap ? (
              <>
                <span>Featured match</span>
                <span>{liveMap.name}</span>
                <span>{Object.values(live.lineup).map((bot) => names[bot] ?? bot).join(" vs ")}</span>
                <span>{LIVE_SPEED}× speed</span>
              </>
            ) : (
              <span>No matches played yet</span>
            )}
          </p>
          <div className="overview-actions">
            <span className="button button-play" aria-disabled="true" title="Playing against the bots in the browser is coming soon">
              <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
                <path d="M5 2.5 L19 11 L5 19.5 Z" />
              </svg>
              <span>
                Play game
                <small>Coming soon</small>
              </span>
            </span>
            <Link href={live ? `/games/${game.id}/matches/${live.id}` : `/games/${game.id}/watch`} className="button button-ghost button-watch">
              <svg width="26" height="22" viewBox="0 0 26 22" aria-hidden="true">
                <rect x="1.5" y="1.5" width="23" height="19" rx="2.5" fill="none" stroke="currentColor" strokeWidth="2" />
                <path d="M10.5 6.5 L17 11 L10.5 15.5 Z" fill="currentColor" />
              </svg>
              Watch a bot match
            </Link>
          </div>
        </section>
        <Ladder gameId={game.id} tables={tables} note={note} />
      </div>

      {more.length > 0 && (
        <section className="more-matches">
          <div className="section-title">
            <h2>More bot matches</h2>
            <Link href={`/games/${game.id}/watch`}>All replays</Link>
          </div>
          <div className="match-grid">
            {more.map((row) => (
              <MatchCard key={row.id} game={game} row={row} botNames={names} />
            ))}
          </div>
        </section>
      )}
    </>
  );
}
