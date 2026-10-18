"use client";

// Play a game against the bots, or watch bots play each other, in the browser. Shared by
// every game: the setup (map, seat, bots), the scoreboard, the clock, full screen and the
// result. The game brings its board and the controls that lie over it (GameView.play);
// Python runs the match in a worker (usePlayMatch).

import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type FormEvent, type ReactNode, type RefObject } from "react";

import { gameView } from "@/games/index.ts";
import { describeOutcome, formatClock, listNames } from "@/lib/format.ts";
import { PERSON, drawnTick, lineupFor, playsOn, result } from "@/lib/play.ts";
import type { GameData } from "@/lib/types.ts";

import { BotAvatar } from "./BotAvatar.tsx";
import { MapPreview } from "./MapPreview.tsx";
import { Scoreboard } from "./Scoreboard.tsx";
import { SpeedSelect } from "./SpeedSelect.tsx";
import { useTheme } from "./ThemedBoard.tsx";
import { typing, useAnimationClock } from "./usePlayback.ts";
import { usePlayMatch, type Setup } from "./usePlayMatch.ts";

const SPEEDS = [0.5, 1, 2, 4];
// A bot that thinks this long a move on the tournament's machines may slow a browser game down.
const SLOW_MS = 20;

type Props = {
  game: GameData;
  /** Each bot's rating, by match size ("2", "3"), from the last tournament. */
  ratings: Record<string, Record<string, number>>;
  /** How long each bot thought a move in the last tournament, in milliseconds. */
  thinking: Record<string, number>;
};

/** `watch`: every seat is a bot, and the person watches. */
type Choice = { map: string; seat: string; watch: boolean; bots: Record<string, string> };

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
  const sized = ratings[String(map.seats.length)] ?? {};
  // A bot that sits out this map's group (Graphite's large maps, say) isn't offered for it.
  const bots = game.bots.filter((bot) => playsOn(bot, map)).sort((a, b) => (sized[a.id] ?? Infinity) - (sized[b.id] ?? Infinity));
  const benched = game.bots.filter((bot) => !playsOn(bot, map)).map((bot) => bot.name);
  const watching = choice.watch;
  const lineup = lineupFor(map.seats, watching ? null : choice.seat, choice.bots, bots.map((bot) => bot.id), fallbackBot);
  const limit = (map.max_ticks ?? game.max_ticks) * game.tick_seconds;
  // The main maps by match size, then each group of maps under its own heading.
  const mapGroups = [...new Set(game.maps.map((entry) => entry.group || `${entry.seats.length} players`))].sort(
    (a, b) => Number(!a.endsWith(" players")) - Number(!b.endsWith(" players")) || a.localeCompare(b),
  );

  const start = (event?: FormEvent) => {
    event?.preventDefault();
    const next = { map: map.id, lineup, seed: Math.floor(Math.random() * 2 ** 31) };
    setSetup(next);
    live.start(next);
    live.speed(speed);
  };

  const inMatch = setup !== null && !["loading", "broken", "ready"].includes(live.phase) && !(live.phase === "failed" && !live.match);
  if (!inMatch) {
    const slow = [...new Set(map.seats.filter((seat) => lineup[seat] !== PERSON && (thinking[lineup[seat]] ?? 0) > SLOW_MS).map((seat) => names[lineup[seat]]))];
    return (
      <form className="play-setup" onSubmit={start}>
        <div className="play-setup-map">
          <MapPreview game={game} map={map} />
        </div>
        <div className="play-setup-fields">
          <div className="play-mode" role="group" aria-label="Play or watch">
            <button type="button" aria-pressed={!watching} onClick={() => setChoice({ ...choice, watch: false })}>
              Play against bots
            </button>
            <button type="button" aria-pressed={watching} onClick={() => setChoice({ ...choice, watch: true })}>
              Watch bots play
            </button>
          </div>
          <label>
            Map
            <select value={map.id} onChange={(event) => setChoice({ ...choice, map: event.target.value })}>
              {mapGroups.map((group) => (
                <optgroup key={group} label={group}>
                  {game.maps
                    .filter((entry) => (entry.group || `${entry.seats.length} players`) === group)
                    .map((entry) => (
                      <option key={entry.id} value={entry.id}>
                        {entry.name}
                        {entry.group ? ` (${entry.seats.length} players)` : ""}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
          </label>
          {!watching && (
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
          )}
          {map.seats
            .filter((seat) => lineup[seat] !== PERSON)
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
            {watching ? "Watch" : "Play"}
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
          {(limit !== game.max_ticks * game.tick_seconds || benched.length > 0) && (
            <p className="small">
              {limit !== game.max_ticks * game.tick_seconds && `A match on ${map.name} lasts up to ${limit % 60 ? formatClock(limit) : `${limit / 60} minutes`}. `}
              {benched.length > 0 && `${listNames(benched)} ${benched.length === 1 ? "sits" : "sit"} out the ${(map.group ?? "").toLowerCase()}.`}
            </p>
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
  const [theme] = useTheme(game.id, game.themes);
  const { phase, match, pause, resume } = live;
  const seatInfo = Object.fromEntries(game.seats.map((seat) => [seat.id, seat]));
  const seats = Object.keys(setup.lineup);
  const person = seats.find((seat) => setup.lineup[seat] === PERSON) ?? null; // null while watching bots
  const tickMs = (game.tick_seconds * 1000) / speed;

  // Redraw with the screen while the match runs, so fleets glide between ticks.
  const [now, setNow] = useState(0);
  useAnimationClock(phase === "playing", () => setNow(performance.now()));

  // The board is brought into view as it is set out, so the whole map shows while playing.
  const area = useRef<HTMLDivElement>(null);
  const [tools, setTools] = useState<HTMLElement | null>(null); // where the game's own buttons go
  const layer = useRef<SVGSVGElement>(null); // the game's controls, over the board
  const [full, toggleFull] = useFullScreen(area);
  useEffect(() => {
    if (phase === "countdown" && !full) area.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [phase, full]);

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
  useEffect(() => {
    if (phase === "countdown" && count <= 0) resume();
  }, [phase, count, resume]);

  const pauseKey = play.pauseKey ?? " ";
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // Fields and shortcuts keep their keys, and a focused button or link keeps its own use of the space bar.
      if (typing(event)) return;
      if (event.key === "f" || event.key === "F") {
        event.preventDefault();
        toggleFull();
        return;
      }
      if (event.key.toLowerCase() !== pauseKey.toLowerCase()) return;
      if (pauseKey === " " && event.target instanceof Element && event.target.closest("button, a, [role=button]")) return;
      event.preventDefault();
      if (phase === "playing") pause();
      else if (phase === "paused") resume();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, pause, resume, pauseKey, toggleFull]);

  // A person's match pauses while the page is hidden (another tab, a minimised window); bots being watched carry on.
  useEffect(() => {
    if (!person || phase !== "playing") return;
    const away = () => {
      if (document.hidden) pause();
    };
    away(); // a match that started while the page was hidden waits too
    document.addEventListener("visibilitychange", away);
    return () => document.removeEventListener("visibilitychange", away);
  }, [person, phase, pause]);

  // A refused move is shown for a moment.
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!live.refusal) return;
    const timer = setTimeout(() => setTick((value) => value + 1), 2600);
    return () => clearTimeout(timer);
  }, [live.refusal]);
  const refusal = live.refusal && performance.now() - live.refusal.at < 2500 ? live.refusal.reason : null;

  // The frame stays while the next match is set out ("Play again"), so full screen, or the cover over the page, holds.
  const frame = (content: ReactNode) => (
    <div className={`play${full === "page" ? " is-full" : ""}`} ref={area} style={{ "--seat": person ? seatInfo[person]?.color : undefined } as CSSProperties}>
      {content}
    </div>
  );
  if (!match) {
    return frame(
      <div className="play-loading" aria-live="polite">
        {live.progress}…
      </div>,
    );
  }

  const position = drawnTick(match.tick, match.playedAt, now, tickMs, phase === "playing");
  const Board = view.Board;
  const Controls = play.Controls;
  const outcome = match.outcome;
  const brokenBots = Object.entries(match.errors).filter(([seat]) => seat !== person);

  return frame(
    <>
      <Scoreboard
        gameId={game.id}
        lineup={setup.lineup}
        seats={game.seats}
        botNames={names}
        stats={view.stats(match.board, match.snapshot)}
        ranking={outcome?.ranking ?? null}
        you={person}
        seconds={match.tick * game.tick_seconds}
        note={`of ${formatClock((game.maps.find((entry) => entry.id === setup.map)?.max_ticks ?? game.max_ticks) * game.tick_seconds)}`}
      />

      <div className="replay-board play-board">
        <Board board={match.board} snapshot={match.snapshot} tick={position} tickSeconds={game.tick_seconds} theme={theme} seats={game.seats} />
        {person && (
          <Controls
            board={match.board}
            snapshot={match.snapshot}
            theme={theme}
            seat={person}
            person={match.people[person]}
            active={phase === "playing"}
            send={(command) => live.command(person, command)}
            tools={tools}
            layer={{ ref: layer, viewBox: play.viewBox(match.board).join(" "), className: "play-controls-layer" }}
            toBoard={(event) => {
              const matrix = layer.current?.getScreenCTM();
              return matrix ? new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse()) : null;
            }}
          />
        )}
        {phase === "countdown" && count > 0 && (
          <div className="play-overlay play-countdown" aria-live="assertive">
            {count}
          </div>
        )}
        {phase === "paused" && (
          <button type="button" className="play-overlay play-paused" onClick={live.resume}>
            Paused. Click or press {keyName(pauseKey)} to carry on.
          </button>
        )}
        {outcome && (
          <div className="replay-result play-result">
            <p>
              {person && <strong>{{ won: "You won!", lost: "You lost.", drew: "A draw." }[result(outcome, person)]} </strong>}
              {describeOutcome(outcome, setup.lineup, names, seatInfo, game.tick_seconds)}
            </p>
            <div className="play-result-actions">
              <button type="button" onClick={again}>
                {person ? "Play again" : "Watch again"}
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
        <SpeedSelect speeds={SPEEDS} value={speed} onChange={onSpeed} />
        <span className="play-tools" ref={setTools} />
        <button type="button" onClick={toggleFull} aria-pressed={full !== false} title="Full screen (F)">
          {full ? "Exit full screen" : "Full screen"}
        </button>
        <button type="button" onClick={live.leave}>
          {outcome || phase === "failed" ? "New game" : person ? "Give up" : "Stop"}
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
        {person &&
          play.help.map((line) => (
            <li key={line}>{line}</li>
          ))}
        <li>{keyName(pauseKey)} pauses and carries on. F switches full screen on and off.</li>
      </ul>
    </>,
  );
}

/** A key as the help names it: "Space", "Esc", "P". */
function keyName(key: string): string {
  return ({ " ": "Space", Escape: "Esc" } as Record<string, string>)[key] ?? key.toUpperCase();
}

/**
 * Full screen for the match: the browser's own where it has one ("screen"; Escape leaves it),
 * else the match covers the page ("page"), as on phones whose browsers keep full screen for
 * videos. Leaving the match leaves full screen.
 */
function useFullScreen(area: RefObject<HTMLElement | null>) {
  const [full, setFull] = useState<false | "screen" | "page">(false);
  useEffect(() => {
    const onChange = () => setFull((now) => (document.fullscreenElement ? "screen" : now === "screen" ? false : now));
    document.addEventListener("fullscreenchange", onChange);
    return () => {
      document.removeEventListener("fullscreenchange", onChange);
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    };
  }, []);
  useEffect(() => {
    if (full !== "page") return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFull(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [full]);
  const toggle = useCallback(() => {
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else if (full === "page") {
      setFull(false);
    } else if (document.fullscreenEnabled && area.current?.requestFullscreen) {
      area.current.requestFullscreen().catch(() => setFull("page"));
    } else {
      setFull("page");
    }
  }, [area, full]);
  return [full, toggle] as const;
}

/** The last setup played on this device, so "play again later" starts where it left off. */
function useStoredChoice(game: GameData, preferredMap: string | undefined) {
  const key = `arena-play-${game.id}`;
  const first = game.maps.find((entry) => entry.id === preferredMap) ?? game.maps[0];
  const [choice, setChoice] = useState<Choice>({ map: first.id, seat: first.seats[0], watch: false, bots: {} });
  // Only maps, seats and bots that still exist: a saved choice may be older than them.
  const settle = (next: Choice): Choice => {
    const map = game.maps.find((entry) => entry.id === next.map) ?? first;
    const seat = map.seats.includes(next.seat) ? next.seat : map.seats[0];
    const known = new Set(game.bots.map((bot) => bot.id));
    return { map: map.id, seat, watch: next.watch === true, bots: Object.fromEntries(Object.entries(next.bots ?? {}).filter(([, bot]) => known.has(bot))) };
  };
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(key) ?? "null") as Choice | null;
      if (saved) setChoice(settle(saved));
    } catch {
      // Nothing saved, or no storage here: the defaults stand.
    }
    // ?watch, as the Watch page's text links here: set up a match between bots.
    if (new URLSearchParams(window.location.search).has("watch")) setChoice((now) => ({ ...now, watch: true }));
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
