// A tournament's standings for one match size, and who beat whom.

import Link from "next/link";

import { percent } from "@/lib/format.ts";
import type { Table } from "@/lib/types.ts";

import { BotAvatar } from "./BotAvatar.tsx";
import { BotName } from "./BotName.tsx";

export function LeaderboardTable({ table, gameId, limit }: { table: Table; gameId: string; limit?: number }) {
  const entries = limit ? table.entries.slice(0, limit) : table.entries;
  const twoPlayers = table.players === 2;
  return (
    <div className="table-scroll">
      <table className="leaderboard">
        <thead>
          <tr>
            <th className="numeric">#</th>
            <th>Bot</th>
            <th className="numeric" title="Elo scale: 100 points more wins about 64% of games">Rating</th>
            {!limit && <th className="numeric">Games</th>}
            {!limit && <th className="numeric">{twoPlayers ? "Won · drawn · lost" : "First · shared · other"}</th>}
            <th className="numeric" title={twoPlayers ? "A win counts 1, a draw 1/2" : "First place counts 1, last 0"}>
              Score
            </th>
            {!limit && <th className="numeric">Think time</th>}
            {!limit && <th className="numeric" title="Moves that raised an error or that the game refused">Faults</th>}
          </tr>
        </thead>
        <tbody>
          {entries.map((entry, index) => (
            <tr key={entry.bot}>
              <td className="numeric rank">{index + 1}</td>
              <td>
                <Link href={`/games/${gameId}/bots#${entry.bot}`} className="bot-cell">
                  <BotAvatar id={entry.bot} kind={entry.kind} size={28} />
                  <BotName name={entry.name} />
                  {entry.kind === "neural" && !limit && <span className="tag">neural</span>}
                </Link>
              </td>
              <td className="numeric">
                <strong>{entry.rating}</strong>
                <span className="interval"> ±{entry.interval}</span>
              </td>
              {!limit && <td className="numeric">{entry.games}</td>}
              {!limit && (
                <td className="numeric">
                  {entry.wins} · {entry.draws} · {entry.losses}
                </td>
              )}
              <td className="numeric">{limit ? percent(entry.score) : <ScoreBar value={entry.score} />}</td>
              {!limit && <td className="numeric">{formatThink(entry.think_ms_mean)}</td>}
              {!limit && <td className="numeric">{entry.errors + entry.refused}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatThink(ms: number): string {
  if (ms < 0.1) return "<0.1 ms";
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  return `${Math.round(ms)} ms`;
}

export function ScoreBar({ value }: { value: number }) {
  return (
    <span className="score">
      <span className="score-bar" aria-hidden="true">
        <span style={{ width: percent(value) }} />
      </span>
      {percent(value)}
    </span>
  );
}

export function HeadToHead({ table }: { table: Table }) {
  const bots = table.entries;
  return (
    <div className="table-scroll">
      <table className="head-to-head">
        <thead>
          <tr>
            <th>Row against column</th>
            {bots.map((bot) => (
              <th key={bot.bot} title={bot.name} className="head-to-head-column">
                <BotAvatar id={bot.bot} kind={bot.kind} size={24} />
                <span>{bot.name}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bots.map((row) => (
            <tr key={row.bot}>
              <th scope="row">{row.name}</th>
              {bots.map((column) => {
                if (row.bot === column.bot) return <td key={column.bot} className="self" />;
                const cell = table.head_to_head[row.bot]?.[column.bot];
                if (!cell || !cell.games) return <td key={column.bot} className="empty">·</td>;
                const share = cell.points / cell.games;
                return (
                  <td
                    key={column.bot}
                    style={{ background: `color-mix(in oklab, var(--win) ${Math.round(share * 100)}%, var(--loss))` }}
                    title={`${row.name} against ${column.name}: ${cell.points} of ${cell.games} points`}
                  >
                    {percent(share)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
