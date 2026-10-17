"use client";

// The top of a game's leaderboard on its overview, one match size at a time.

import Link from "next/link";
import { useState } from "react";

import { BotAvatar } from "./BotAvatar.tsx";
import { BotName } from "./BotName.tsx";

export type LadderTable = {
  players: number;
  label: string;
  entries: { bot: string; name: string; kind: string; rating: number }[];
};

export function Ladder({ gameId, tables, note }: { gameId: string; tables: LadderTable[]; note: string }) {
  const [shown, setShown] = useState(0);
  const table = tables[shown];
  return (
    <aside className="ladder" aria-labelledby="ladder-title">
      <div className="ladder-head">
        <h2 id="ladder-title">AI ladder</h2>
        {tables.length > 1 && (
          <div className="ladder-sizes" role="group" aria-label="Match size">
            {tables.map((entry, index) => (
              <button
                key={entry.players}
                type="button"
                aria-pressed={index === shown}
                onClick={() => setShown(index)}
                title={entry.label}
              >
                {entry.players}P
              </button>
            ))}
          </div>
        )}
      </div>
      {table ? (
        <>
          <p className="ladder-label">Rating, {table.label}</p>
          <ol className="ladder-rows">
            {table.entries.map((entry, index) => (
              <li key={entry.bot}>
                <span className="ladder-rank">{index + 1}</span>
                <Link href={`/games/${gameId}/bots#${entry.bot}`} className="ladder-bot">
                  <BotAvatar game={gameId} id={entry.bot} kind={entry.kind} size={32} />
                  <BotName name={entry.name} />
                </Link>
                <span className="ladder-rating">{entry.rating}</span>
              </li>
            ))}
          </ol>
        </>
      ) : (
        <p className="ladder-label">No tournament has been played yet.</p>
      )}
      <Link href={`/games/${gameId}/leaderboard`} className="button button-ghost ladder-all">
        View all bots
        <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
          <path d="M6 3 L11 8 L6 13" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </Link>
      {note && <p className="ladder-note">{note}</p>}
    </aside>
  );
}
