"use client";

// Play a game against the bots, in the browser. Shared by every game: the setup (map,
// seat, opponents), the scoreboard, the clock and the result. The game brings its board
// and the controls that lie over it (GameView.play); Python runs the match in a worker
// (usePlayMatch).

import { useEffect, useMemo, useRef, useState, type CSSProperties, type FormEvent } from "react";

import { gameView } from "@/games/index.ts";
import { describeOutcome, formatClock } from "@/lib/format.ts";
import { PERSON, drawnTick, lineupFor, result } from "@/lib/play.ts";
import type { GameData } from "@/lib/types.ts";

import { BotAvatar } from "./BotAvatar.tsx";
import { BotName } from "./BotName.tsx";
import { MapPreview } from "./MapPreview.tsx";
import { usePlayMatch, type Setup } from "./usePlayMatch.ts";

const SPEEDS = [0.5, 1, 2];
// A bot that thinks this long a move on the tournament's machines may slow a browser game down.
const SLOW_MS = 20;

type Props = {
  game: GameData;
  /** Each bot's rating, by match size ("2", "3"), from the last tournament. */
  ratings: Record<string, Record<string, number>>;
  /** How long each bot thought a move in the last tournament, in milliseconds. */
  thinking: Record<string, number>;
};

type Choice = { map: string; seat: string; bots: Record<string, string> };

export function PlayGame({ game, ratings, thinking }: Props) {
  const view = gameView(game.id);
  const play = view.play!;
  const names = useMemo<Record<string, string>>(() => ({ ...Object.fromEntries(game.bots.map((bot) => [bot.id, bot.name])), [PERSON]: "You" }), [game.bots]);
  const seatInfo = useMemo(() => Object.fromEntries(game.seats.map((seat) => [seat.id, seat])), [game.seats]);
  const fallbackBot = game.rating_anchor?.[0] ?? game.bots[0]?.id ?? "";
  const [choice, setChoice] = useStoredChoice(game, play.map);
  const [speed, setSpeed] = useState(1);
  const [setup, setSetup] = useState<Setup | null>(null);
  const live = usePlayMatch(game.id, game.tick_seconds);

  const map = game.maps.find((entry) => entry.id === choice.map) ?? game.maps[0];
  const lineup = lineupFor(map.seats, choice.seat, choice.bots, fallbackBot);
  const sized = ratings[String(map.seats.length)] ?? {};
  const bots = [...game.bots].sort((a, b) => (sized[a.id] ?? Infinity) - (sized[b.id] ?? Infinity));

  const start = (event?: FormEvent) => {
    event?.preventDefault();
    const next = { map: map.id, lineup, seed: Math.floor(Math.random() * 2 ** 31) };
    setSetup(next);
    live.start(next);
    live.speed(speed);
  };

  const inMatch = setup !== null && !["loading", "broken", "ready"].includes(live.phase) && !(live.phase === "failed" && !live.match);
  if (!inMatch) {
    const slow = map.seats.filter((seat) => seat !== choice.seat && (thinking[lineup[seat]] ?? 0) > SLOW_MS).map((seat) => names[lineup[seat]]);
    return (
      <form className="play-setup" onSubmit={start}>
        <div className="play-setup-map">
          <MapPreview game={game} map={map} />
        </div>
        <div className="play-setup-fields">
          <label>
            Map
            <select value={map.id} onChange={(event) => setChoice({ ...choice, map: event.target.value })}>
              {[...new Set(game.maps.map((entry) => entry.seats.length))].map((size) => (
                <optgroup key={size} label={`${size} players`}>
                  {game.maps
                    .filter((entry) => entry.seats.length === size)
                    .map((entry) => (
                      <option key={entry.id} value={entry.id}>
                        {entry.name}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
          </label>
          <label>
            You play
            <select value={choice.seat} onChange={(event) => setChoice({ ...choice, seat: event.target.value })}>
              {map.seats.map((seat) => (
                <option key={seat} value={seat}>
                  {seatInfo[seat]?.name ?? seat}
                </option>
              ))}
            </select>
          </label>
          {map.seats
            .filter((seat) => seat !== choice.seat)
            .map((seat) => (
              <label key={seat} style={{ "--seat": seatInfo[seat]?.color } as CSSProperties} className="play-setup-opponent">
                {seatInfo[seat]?.name ?? seat}
                <span className="play-setup-bot">
                  <BotAvatar game={game.id} id={lineup[seat]} size={34} />
                  <select value={lineup[seat]} onChange={(event) => setChoice({ ...choice, bots: { ...choice.bots, [seat]: event.target.value } })}>
                    {bots.map((bot) => (
                      <option key={bot.id} value={bot.id}>
                        {bot.name}
                        {sized[bot.id] !== undefined ? ` (rated ${sized[bot.id]})` : ""}
                      </option>
                    ))}
                  </select>
                </span>
              </label>
            ))}
          <button type="submit" className="button-primary play-start" disabled={live.phase === "broken"}>
            Play
          </button>
          <p className="small play-status" aria-live="polite">
            {live.phase === "loading"
              ? `${live.progress}… The first time, the browser fetches about 15 MB of Python; after that it is quick.`
              : live.phase === "broken"
                ? `This browser could not run the game: ${live.error}`
                : live.phase === "failed"
                  ? `The game could not start: ${live.error}`
                  : "Ready."}
          </p>
          {live.phase === "broken" && (
            <button type="button" className="play-reload" onClick={() => window.location.reload()}>
              Reload the page to try again
            </button>
          )}
          {slow.length > 0 && (
            <p className="small">
              {slow.join(" and ")} {slow.length === 1 ? "thinks" : "think"} hard about every move. In a browser the game may slow down to
              wait for {slow.length === 1 ? "it" : "them"}.
            </p>
          )}
        </div>
      </form>
    );
  }

  return (
    <Match
      game={game}
      setup={setup!}
      names={names}
      speed={speed}
      onSpeed={(value) => {
        setSpeed(value);
        live.speed(value);
      }}
      live={live}
      again={() => start()}
    />
  );
}

function Match({
  game,
  setup,
  names,
  speed,
  onSpeed,
  live,
  again,
}: {
  game: GameData;
  setup: Setup;
  names: Record<string, string>;
  speed: number;
  onSpeed: (speed: number) => void;
  live: ReturnType<typeof usePlayMatch>;
  again: () => void;
}) {
  const view = gameView(game.id);
  const play = view.play!;
  const { phase, match } = live;
  const seatInfo = Object.fromEntries(game.seats.map((seat) => [seat.id, seat]));
  const seats = Object.keys(setup.lineup);
  const person = seats.find((seat) => setup.lineup[seat] === PERSON) ?? seats[0];
  const tickMs = (game.tick_seconds * 1000) / speed;

  // Redraw with the screen while the match runs, so fleets glide between ticks.
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (phase !== "playing") return;
    let frame = requestAnimationFrame(function draw(time) {
      setNow(time);
      frame = requestAnimationFrame(draw);
    });
    return () => cancelAnimationFrame(frame);
  }, [phase]);

  // The board is brought into view as it is set out, so the whole map shows while playing.
  const area = useRef<HTMLDivElement>(null);
  const [tools, setTools] = useState<HTMLElement | null>(null); // where the game's own buttons go
  useEffect(() => {
    if (phase === "countdown") area.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [phase]);

  // 3, 2, 1, then the clock starts.
  const [count, setCount] = useState(3);
  useEffect(() => {
    // Wound back while there is no countdown, so the next one (after "Play again") starts at 3.
    if (phase !== "countdown") {
      setCount(3);
      return;
    }
    const timer = setInterval(() => setCount((value) => value - 1), 600);
    return () => clearInterval(timer);
  }, [phase]);
  const { go } = live;
  useEffect(() => {
    if (phase === "countdown" && count <= 0) go();
  }, [phase, count, go]);

  const { pause, resume } = live;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // A focused button, link or field keeps its own use of the space bar.
      if (event.key !== " " || (event.target instanceof Element && event.target.closest("button, a, input, select, textarea, [role=button]"))) return;
      event.preventDefault();
      if (phase === "playing") pause();
      else if (phase === "paused") resume();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, pause, resume]);

  // A refused move is shown for a moment.
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!live.refusal) return;
    const timer = setTimeout(() => setTick((value) => value + 1), 2600);
    return () => clearTimeout(timer);
  }, [live.refusal]);
  const refusal = live.refusal && performance.now() - live.refusal.at < 2500 ? live.refusal.reason : null;

  if (!match) {
    return (
      <div className="play-loading" aria-live="polite">
        {phase === "failed" ? (
          <>
            <p>The game could not start: {live.error}</p>
            <button type="button" onClick={live.leave}>
              Back
            </button>
          </>
        ) : (
          <p>{live.progress}…</p>
        )}
      </div>
    );
  }

  const position = drawnTick(match.tick, match.playedAt, now, tickMs, phase === "playing");
  const stats = view.stats(match.board, match.snapshot);
  const Board = view.Board;
  const Controls = play.Controls;
  const outcome = match.outcome;
  const brokenBots = Object.entries(match.errors).filter(([seat]) => seat !== person);

  return (
    <div className="play" ref={area} style={{ "--seat": seatInfo[person]?.color } as CSSProperties}>
      <div className="scoreboard" style={{ "--seats": seats.length } as CSSProperties}>
        {seats.map((seat) => {
          const out = outcome !== null && outcome.ranking.findIndex((group) => group.includes(seat)) > 0;
          return (
            <div key={seat} className={`scoreboard-seat${out ? " is-out" : ""}${seat === person ? " is-you" : ""}`} style={{ "--seat": seatInfo[seat]?.color } as CSSProperties}>
              <span className="scoreboard-name">{seat === person ? "You" : <BotName name={names[setup.lineup[seat]] ?? setup.lineup[seat]} game={game.id} id={setup.lineup[seat]} />}</span>
              <span className="scoreboard-color">{seatInfo[seat]?.name ?? seat}</span>
              <span className="scoreboard-stats">
                {stats[seat] ? (
                  stats[seat].map((stat) => (
                    <span key={stat.label}>
                      <strong>{stat.value}</strong> {stat.label}
                    </span>
                  ))
                ) : (
                  <span>Knocked out</span>
                )}
              </span>
            </div>
          );
        })}
        <div className="scoreboard-clock">
          <span>{formatClock(match.tick * game.tick_seconds)}</span>
          <small>of {formatClock(game.max_ticks * game.tick_seconds)}</small>
        </div>
      </div>

      <div className="replay-board play-board">
        <Board board={match.board} snapshot={match.snapshot} tick={position} tickSeconds={game.tick_seconds} theme={game.theme} seats={game.seats} />
        <Controls
          board={match.board}
          snapshot={match.snapshot}
          theme={game.theme}
          seat={person}
          person={match.people[person]}
          active={phase === "playing"}
          send={(command) => live.command(person, command)}
          tools={tools}
        />
        {phase === "countdown" && (
          <div className="play-overlay play-countdown" aria-live="assertive">
            {count > 0 ? count : "Go"}
          </div>
        )}
        {phase === "paused" && (
          <button type="button" className="play-overlay play-paused" onClick={live.resume}>
            Paused. Click or press space to carry on.
          </button>
        )}
        {outcome && (
          <div className="replay-result play-result">
            <p>
              <strong>{{ won: "You won!", lost: "You lost.", drew: "A draw." }[result(outcome, person)]}</strong>{" "}
              {describeOutcome(outcome, setup.lineup, names, seatInfo, game.tick_seconds)}
            </p>
            <div className="play-result-actions">
              <button type="button" onClick={again}>
                Play again
              </button>
              <button type="button" onClick={live.leave}>
                Change
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="replay-controls play-controls">
        {phase === "paused" ? (
          <button type="button" className="button-primary" onClick={live.resume}>
            Carry on
          </button>
        ) : (
          <button type="button" className="button-primary" onClick={live.pause} disabled={phase !== "playing"}>
            Pause
          </button>
        )}
        <label className="replay-speed">
          Speed
          <select value={speed} onChange={(event) => onSpeed(Number(event.target.value))}>
            {SPEEDS.map((value) => (
              <option key={value} value={value}>
                {value}×
              </option>
            ))}
          </select>
        </label>
        <span className="play-tools" ref={setTools} />
        <button type="button" onClick={live.leave}>
          {outcome || phase === "failed" ? "New game" : "Give up"}
        </button>
        <span className="play-notes" aria-live="polite">
          {refusal ? (
            <span className="play-refused">Not sent: {refusal}</span>
          ) : phase === "playing" && match.pace < 0.85 ? (
            `The bots are thinking hard, so the game runs at ${Math.round(match.pace * 100)}% speed.`
          ) : null}
        </span>
      </div>
      {phase === "failed" && <p className="notice">The game stopped because of an error: {live.error}</p>}
      {brokenBots.map(([seat, error]) => (
        <p key={seat} className="notice small">
          {names[setup.lineup[seat]]} ({seatInfo[seat]?.name ?? seat}) stopped working and sits the rest out: {error}
        </p>
      ))}
      <ul className="play-help">
        {play.help.map((line) => (
          <li key={line}>{line}</li>
        ))}
        <li>Space pauses and carries on.</li>
      </ul>
    </div>
  );
}

/** The last setup played on this device, so "play again later" starts where it left off. */
function useStoredChoice(game: GameData, preferredMap: string | undefined) {
  const key = `arena-play-${game.id}`;
  const first = game.maps.find((entry) => entry.id === preferredMap) ?? game.maps[0];
  const [choice, setChoice] = useState<Choice>({ map: first.id, seat: first.seats[0], bots: {} });
  // Only maps, seats and bots that still exist: a saved choice may be older than them.
  const settle = (next: Choice): Choice => {
    const map = game.maps.find((entry) => entry.id === next.map) ?? first;
    const seat = map.seats.includes(next.seat) ? next.seat : map.seats[0];
    const known = new Set(game.bots.map((bot) => bot.id));
    return { map: map.id, seat, bots: Object.fromEntries(Object.entries(next.bots ?? {}).filter(([, bot]) => known.has(bot))) };
  };
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(key) ?? "null") as Choice | null;
      if (saved) setChoice(settle(saved));
    } catch {
      // Nothing saved, or no storage here: the defaults stand.
    }
  }, [key]); // once, when the page opens
  const update = (next: Choice) => {
    const settled = settle(next);
    setChoice(settled);
    try {
      localStorage.setItem(key, JSON.stringify(settled));
    } catch {
      // Not saved; the choice still holds for this visit.
    }
  };
  return [choice, update] as const;
}
