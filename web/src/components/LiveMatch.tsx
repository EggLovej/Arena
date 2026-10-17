"use client";

// The match on a game's overview: a featured replay that plays by itself and
// starts again, framed like a screen, with a heads-up display made from the
// game's own scoreboard numbers. It works for every game.

import { useEffect, useRef, useState, type CSSProperties } from "react";

import { gameView } from "@/games/index.ts";
import { describeOutcome, formatClock } from "@/lib/format.ts";
import type { MapEntry, Seat } from "@/lib/types.ts";

import { BotName } from "./BotName.tsx";
import { useAnimationClock, useReplay } from "./usePlayback.ts";

const HOLD_SECONDS = 5; // how long the result shows before the match starts again
const LEAD_SECONDS = 3; // how long before the first move the loop starts

type Props = {
  gameId: string;
  gameName: string;
  replayUrl: string;
  map: MapEntry; // drawn until the replay has loaded
  lineup: Record<string, string>; // seat to bot
  seats: Seat[];
  botNames: Record<string, string>;
  theme: unknown;
  tickSeconds: number;
  speed: number;
};

export function LiveMatch({ gameId, gameName, replayUrl, map, lineup, seats, botNames, theme, tickSeconds, speed }: Props) {
  const view = gameView(gameId);
  const { replay, snapshots } = useReplay(replayUrl);
  const [position, setPosition] = useState(0);
  const [paused, setPaused] = useState(false);
  const [visible, setVisible] = useState(true);
  const stage = useRef<HTMLDivElement>(null);

  // Moving pictures wait for a click when the viewer asked for less motion.
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) setPaused(true);
  }, []);

  // Nothing moves while the board is scrolled out of sight.
  useEffect(() => {
    const element = stage.current;
    if (!element) return;
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const last = Math.max(0, snapshots.length - 1);
  // Bots sometimes wait a long time before their first move, so the loop starts just before it.
  const firstMove = replay?.actions[0]?.[0] ?? 0;
  const start = Math.min(last, Math.max(0, firstMove - LEAD_SECONDS / tickSeconds));
  useEffect(() => setPosition(start), [start]);

  const finished = replay !== null && position >= last;
  // A long gap (the tab was hidden) carries on where it was rather than jumping ahead.
  useAnimationClock(replay !== null && !paused && visible && !finished, (seconds) =>
    setPosition((current) => Math.min(last, current + (Math.min(seconds, 0.25) / tickSeconds) * speed)),
  );

  // The result stays up while the board is out of sight, so it is still there on scrolling back.
  useEffect(() => {
    if (!finished || paused || !visible) return;
    const timer = setTimeout(() => setPosition(start), HOLD_SECONDS * 1000);
    return () => clearTimeout(timer);
  }, [finished, paused, visible, start]);

  const board = replay?.board ?? map.board;
  const tick = Math.max(0, Math.min(last, Math.floor(position)));
  const snapshot = replay ? snapshots[tick] : map.start;
  const stats = view.stats(board, snapshot);
  const strength = view.strength(board, snapshot);
  const total = Object.values(strength).reduce((sum, value) => sum + value, 0);
  const seatInfo = Object.fromEntries(seats.map((seat) => [seat.id, seat]));
  const order = Object.keys(lineup);
  const half = Math.ceil(order.length / 2);
  const Board = view.Board;

  const panel = (seat: string, right: boolean) => (
    <div key={seat} className={`hud-seat${right ? " is-right" : ""}`} style={{ "--seat": seatInfo[seat]?.color } as CSSProperties}>
      <span className="hud-name">
        <BotName name={botNames[lineup[seat]] ?? lineup[seat]} />
      </span>
      <span className="hud-stats">
        {stats[seat] ? stats[seat].map((stat) => <span key={stat.label}>{`${stat.value} ${stat.label}`}</span>) : <span>knocked out</span>}
      </span>
      <span className="hud-bar" aria-hidden="true">
        <span style={{ width: `${total > 0 ? ((strength[seat] ?? 0) / total) * 100 : 0}%` }} />
      </span>
    </div>
  );

  return (
    <div className="stage" ref={stage}>
      <div className="hud">
        {order.slice(0, half).map((seat) => panel(seat, false))}
        <div className="hud-clock">
          <span>{gameName}</span>
          <strong>{formatClock(position * tickSeconds)}</strong>
        </div>
        {order.slice(half).map((seat) => panel(seat, true))}
      </div>
      <div className="stage-board">
        <Board board={board} snapshot={snapshot} tick={replay ? position : 0} tickSeconds={tickSeconds} theme={theme} seats={seats} />
        {finished && replay && (
          <div className="replay-result">
            <p>{describeOutcome(replay.outcome, lineup, botNames, seatInfo, tickSeconds)}</p>
          </div>
        )}
        {replay && (
          <button
            type="button"
            className="stage-toggle"
            onClick={() => setPaused(!paused)}
            aria-label={paused ? "Play the match" : "Pause the match"}
            title={paused ? "Play" : "Pause"}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              {paused ? <path d="M3 1.5 L12 7 L3 12.5 Z" /> : <path d="M2.5 1.5 h3 v11 h-3 Z M8.5 1.5 h3 v11 h-3 Z" />}
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
