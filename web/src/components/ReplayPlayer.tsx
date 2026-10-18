"use client";

// Plays a recorded match back: the game's own board, a scoreboard, a chart of
// how each side did over time, and controls to pause, seek and change speed.
// It works for every game; only the board and the scoreboard numbers differ.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { gameView } from "@/games/index.ts";
import { describeOutcome, formatClock } from "@/lib/format.ts";
import type { Seat, Snapshot, Theme } from "@/lib/types.ts";

import { Scoreboard } from "./Scoreboard.tsx";
import { SpeedSelect } from "./SpeedSelect.tsx";
import { useTheme } from "./ThemedBoard.tsx";
import { typing, useAnimationClock, useReplay } from "./usePlayback.ts";

const SPEEDS = [0.5, 1, 2, 4, 8, 16];

type Props = {
  gameId: string;
  replayUrl: string;
  seats: Seat[];
  botNames: Record<string, string>;
  themes: Theme[];
};

export function ReplayPlayer({ gameId, replayUrl, seats, botNames, themes }: Props) {
  const view = gameView(gameId);
  const [theme] = useTheme(gameId, themes);
  const { replay, snapshots, error } = useReplay(replayUrl);
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);

  useEffect(() => {
    if (!replay) return;
    setPosition(0);
    setPlaying(true);
  }, [replay]);

  // Once loaded, scroll down to the player if its controls are off screen, so the
  // scoreboard, board and controls fill the window.
  const player = useRef<HTMLDivElement>(null);
  const controls = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const bottom = controls.current?.getBoundingClientRect().bottom;
    if (bottom !== undefined && bottom > window.innerHeight && window.scrollY === 0) player.current?.scrollIntoView({ block: "start" });
  }, [replay]);
  const last = Math.max(0, snapshots.length - 1);
  const tickSeconds = replay?.tick_seconds ?? 0.1;

  useAnimationClock(playing && !!replay, (seconds) => setPosition((current) => Math.min(last, current + (seconds / tickSeconds) * speed)));

  useEffect(() => {
    if (playing && replay && position >= last) setPlaying(false);
  }, [playing, replay, position, last]);

  // These change the position from whatever it is by then, so they stay the same while it plays.
  const move = useCallback((to: (current: number) => number) => setPosition((current) => Math.max(0, Math.min(last, to(current)))), [last]);
  const seek = useCallback((tick: number) => move(() => tick), [move]);
  const toggle = useCallback(() => {
    if (!playing) move((current) => (current >= last ? 0 : current));
    setPlaying(!playing);
  }, [playing, move, last]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (typing(event)) return;
      const second = 1 / tickSeconds;
      if (event.key === " ") {
        event.preventDefault();
        toggle();
      } else if (event.key === "ArrowRight") move((current) => current + (event.shiftKey ? 10 : 1) * second);
      else if (event.key === "ArrowLeft") move((current) => current - (event.shiftKey ? 10 : 1) * second);
      else if (event.key === ".") move((current) => Math.floor(current) + 1);
      else if (event.key === ",") move((current) => Math.ceil(current) - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, move, tickSeconds]);

  // The chart redraws only when these change, not on every frame.
  const seatIds = useMemo(() => replay?.seats.map(({ seat }) => seat) ?? [], [replay]);
  const colors = useMemo(() => Object.fromEntries(seats.map((seat) => [seat.id, seat.color])), [seats]);

  if (error) return <p className="notice">{error}</p>;
  if (!replay) return <div className="replay-loading">Loading the replay…</div>;

  const tick = Math.max(0, Math.min(last, Math.floor(position)));
  const snapshot = snapshots[tick];
  const seatInfo = Object.fromEntries(seats.map((seat) => [seat.id, seat]));
  const lineup = Object.fromEntries(replay.seats.map(({ seat, bot }) => [seat, bot]));
  const finished = tick >= last;
  const Board = view.Board;

  return (
    <div className="replay" ref={player}>
      <Scoreboard
        gameId={gameId}
        lineup={lineup}
        seats={seats}
        botNames={botNames}
        stats={view.stats(replay.board, snapshot)}
        ranking={finished ? replay.outcome.ranking : null}
        seconds={position * tickSeconds}
        note={`tick ${tick}`}
      />

      <div className="replay-board">
        <Board board={replay.board} snapshot={snapshot} next={snapshots[tick + 1]} tick={position} tickSeconds={tickSeconds} theme={theme} seats={seats} />
        {finished && (
          <div className="replay-result">
            <p>{describeOutcome(replay.outcome, lineup, botNames, seatInfo, tickSeconds)}</p>
            <button type="button" onClick={() => { setPosition(0); setPlaying(true); }}>
              Watch again
            </button>
          </div>
        )}
      </div>

      <div className="replay-controls" ref={controls}>
        <button type="button" className="button-primary" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
          {playing ? "Pause" : finished ? "Replay" : "Play"}
        </button>
        <input
          type="range"
          min={0}
          max={last}
          step={1}
          value={tick}
          onChange={(event) => seek(Number(event.target.value))}
          aria-label="Position in the match"
        />
        <span className="replay-time">
          {formatClock(position * tickSeconds)} / {formatClock(last * tickSeconds)}
        </span>
        <SpeedSelect speeds={SPEEDS} value={speed} onChange={setSpeed} />
      </div>
      <StrengthChart
        snapshots={snapshots}
        board={replay.board}
        seats={seatIds}
        colors={colors}
        strength={view.strength}
        label={view.strengthLabel}
        position={position}
        onSeek={seek}
      />
      <p className="replay-keys">Space plays and pauses, ← and → move a second (Shift: ten), comma and full stop move one tick.</p>
    </div>
  );
}

function StrengthChart({
  snapshots,
  board,
  seats,
  colors,
  strength,
  label,
  position,
  onSeek,
}: {
  snapshots: Snapshot[];
  board: unknown;
  seats: string[];
  colors: Record<string, string>;
  strength: (board: unknown, snapshot: Snapshot) => Record<string, number>;
  label: string;
  position: number;
  onSeek: (tick: number) => void;
}) {
  const width = 1000;
  const height = 80;
  const paths = useMemo(() => {
    const step = Math.max(1, Math.floor(snapshots.length / 250));
    const samples: { tick: number; values: Record<string, number> }[] = [];
    for (let tick = 0; tick < snapshots.length; tick += step) samples.push({ tick, values: strength(board, snapshots[tick]) });
    const lastTick = snapshots.length - 1;
    if (samples[samples.length - 1]?.tick !== lastTick) samples.push({ tick: lastTick, values: strength(board, snapshots[lastTick]) });
    const most = Math.max(1, ...samples.flatMap((sample) => seats.map((seat) => sample.values[seat] ?? 0)));
    const x = (tick: number) => (lastTick > 0 ? (tick / lastTick) * width : 0);
    const y = (value: number) => height - 4 - (value / most) * (height - 8);
    return seats.map((seat) => ({
      seat,
      d: samples.map((sample, index) => `${index ? "L" : "M"}${x(sample.tick).toFixed(1)},${y(sample.values[seat] ?? 0).toFixed(1)}`).join(" "),
    }));
  }, [snapshots, board, seats, strength]);

  const lastTick = Math.max(1, snapshots.length - 1);
  return (
    <figure className="strength-chart">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        onClick={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          onSeek(Math.round(((event.clientX - box.left) / box.width) * lastTick));
        }}
        role="img"
        aria-label={label}
      >
        {paths.map(({ seat, d }) => (
          <path key={seat} d={d} fill="none" stroke={colors[seat]} strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
        ))}
        <line x1={(position / lastTick) * width} x2={(position / lastTick) * width} y1={0} y2={height} className="strength-cursor" vectorEffect="non-scaling-stroke" />
      </svg>
      <figcaption>{label}. Click to jump there.</figcaption>
    </figure>
  );
}
