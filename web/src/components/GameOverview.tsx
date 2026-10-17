// A game's front pages, shared by every game: the preview on the home page,
// and the overview on the game's own page. Both show a featured match playing
// on a loop and the top of the ladder.

import Link from "next/link";
import type { CSSProperties } from "react";

import { gameView } from "@/games/index.ts";
import { botNames, getGame, getLeaderboard, getMatches } from "@/lib/data.ts";
import { featuredMatches, liveMatch } from "@/lib/featured.ts";
import type { GameData, MatchRow } from "@/lib/types.ts";

import { Ladder } from "./Ladder.tsx";
import { LeaderboardTable } from "./LeaderboardTable.tsx";
import { LiveMatch } from "./LiveMatch.tsx";
import { MapPreview } from "./MapPreview.tsx";
import { MatchCard } from "./MatchCard.tsx";

const LIVE_SPEED = 2; // unless the game's view sets its own

function load(gameId: string) {
  const game = getGame(gameId);
  const view = gameView(gameId);
  const leaderboard = getLeaderboard(gameId);
  const rows = getMatches(gameId);
  const live = liveMatch(rows, leaderboard, view.liveMap);
  return { game, view, leaderboard, rows, live, names: botNames(game) };
}

function updated(leaderboard: { matches: number; updated: string } | null): string {
  return leaderboard ? `${leaderboard.matches} matches, updated ${new Date(leaderboard.updated).toUTCString().slice(5, 16)}` : "";
}

/** The home page's look at a game: what its own page holds, and the way there. */
export function GamePreview({ gameId }: { gameId: string }) {
  const { game, view, leaderboard, live, names } = load(gameId);
  const tables = Object.values(leaderboard?.tables ?? {}).map((table) => ({
    players: table.players,
    label: table.label,
    entries: table.entries.slice(0, 5).map(({ bot, name, kind, rating }) => ({ bot, name, kind, rating })),
  }));
  return (
    <div style={view.accent ? ({ "--stage-accent": view.accent } as CSSProperties) : undefined}>
      <div className="game-header preview-header">
        <div>
          <h1 className="game-title">
            <Link href={`/games/${game.id}`}>{game.name}</Link>
          </h1>
          <p className="game-tagline">{game.tagline}</p>
        </div>
        <Link href={`/games/${game.id}`} className="button button-ghost preview-open">
          Open {game.name}
          <Arrow />
        </Link>
      </div>
      <div className="overview">
        <section className="overview-stage" aria-label="Featured match">
          <Stage game={game} live={live} names={names} />
          <div className="overview-actions">
            <PlayButton game={game} />
            <WatchButton game={game} live={live} />
          </div>
        </section>
        <Ladder gameId={game.id} tables={tables} note={updated(leaderboard)} />
      </div>
    </div>
  );
}

/** The front of a game's own page: the match, the top bots, and more matches to watch. */
export function GameOverview({ gameId }: { gameId: string }) {
  const { game, leaderboard, rows, live, names } = load(gameId);
  const more = featuredMatches(rows, leaderboard, 7)
    .filter((row) => row.map !== live?.map)
    .slice(0, 6);
  const tables = Object.values(leaderboard?.tables ?? {});
  return (
    <>
      <div className="overview overview-wide-side">
        <section className="overview-stage" aria-label="Featured match">
          <Stage game={game} live={live} names={names} />
          <div className="overview-actions">
            <PlayButton game={game} />
            <WatchButton game={game} live={live} />
          </div>
        </section>
        <aside className="overview-side">
          {tables.map((table) => (
            <div key={table.players}>
              <div className="section-title">
                <h2>Top bots, {table.label}</h2>
                <Link href={`/games/${game.id}/leaderboard`}>Full table</Link>
              </div>
              <LeaderboardTable table={table} gameId={game.id} limit={5} />
            </div>
          ))}
          {leaderboard ? <p className="small">{updated(leaderboard)}.</p> : <p className="notice">No tournament has been played yet.</p>}
        </aside>
      </div>

      {more.length > 0 && (
        <section className="more-matches">
          <div className="section-title">
            <h2>Featured matches</h2>
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

function Stage({ game, live, names }: { game: GameData; live: MatchRow | undefined; names: Record<string, string> }) {
  const map = live && game.maps.find((entry) => entry.id === live.map);
  const view = gameView(game.id);
  const cardMap = game.maps.find((entry) => entry.id === view.cardMap) ?? game.maps[0];
  const speed = view.liveSpeed ?? LIVE_SPEED;
  return (
    <>
      {live && map ? (
        <LiveMatch
          gameId={game.id}
          gameName={game.name}
          replayUrl={`/data/${game.id}/replays/${live.id}.json`}
          map={map}
          lineup={live.lineup}
          seats={game.seats}
          botNames={names}
          theme={game.theme}
          tickSeconds={game.tick_seconds}
          speed={speed}
        />
      ) : (
        <div className="stage">{cardMap && <MapPreview game={game} map={cardMap} />}</div>
      )}
      <p className="stage-legend">
        {live && map ? (
          <>
            <span>Featured match</span>
            <span>{map.name}</span>
            <span>{Object.values(live.lineup).map((bot) => names[bot] ?? bot).join(" vs ")}</span>
            <span>{speed === 1 ? "Real time" : `${speed}× speed`}</span>
          </>
        ) : (
          <span>No matches played yet</span>
        )}
      </p>
    </>
  );
}

function PlayButton({ game }: { game: GameData }) {
  const icon = (
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
      <path d="M5 2.5 L19 11 L5 19.5 Z" />
    </svg>
  );
  if (!gameView(game.id).play) {
    return (
      <span className="button button-play" aria-disabled="true" title="Playing against the bots in the browser is coming soon">
        {icon}
        <span>
          Play game
          <small>Coming soon</small>
        </span>
      </span>
    );
  }
  return (
    <Link href={`/games/${game.id}/play`} className="button button-play">
      {icon}
      <span>
        Play game
        <small>Against the bots</small>
      </span>
    </Link>
  );
}

function WatchButton({ game, live }: { game: GameData; live: MatchRow | undefined }) {
  return (
    <Link href={live ? `/games/${game.id}/matches/${live.id}` : `/games/${game.id}/watch`} className="button button-ghost button-watch">
      <svg width="26" height="22" viewBox="0 0 26 22" aria-hidden="true">
        <rect x="1.5" y="1.5" width="23" height="19" rx="2.5" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M10.5 6.5 L17 11 L10.5 15.5 Z" fill="currentColor" />
      </svg>
      Watch a bot match
    </Link>
  );
}

function Arrow() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M6 3 L11 8 L6 13" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
