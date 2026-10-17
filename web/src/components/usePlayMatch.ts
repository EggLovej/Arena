"use client";

// The play page's hold on its worker (public/workers/play.js), which runs the match in
// Python. Python starts loading as soon as the page opens, while the person picks a map
// and opponents, since that is what takes longest.

import { useCallback, useEffect, useRef, useState } from "react";

import { applyDelta } from "@/lib/frames.ts";
import { PYODIDE_URL, type MatchStart, type MatchUpdate, type Outcome, type WorkerMessage } from "@/lib/play.ts";
import type { Snapshot } from "@/lib/types.ts";

export type Phase = "loading" | "ready" | "starting" | "countdown" | "playing" | "paused" | "over" | "failed";

export type MatchState = {
  board: unknown;
  snapshot: Snapshot;
  tick: number;
  playedAt: number; // performance.now() when the tick arrived
  people: Record<string, unknown>;
  thinking: Record<string, number>;
  errors: Record<string, string>;
  outcome: Outcome | null;
  pace: number; // 1 when the game keeps time; lower when it waits for the bots
};

export type Setup = { map: string; lineup: Record<string, string>; seed: number };

export function usePlayMatch(gameId: string, tickSeconds: number) {
  const worker = useRef<Worker | null>(null);
  const [phase, setPhase] = useState<Phase>("loading");
  const [progress, setProgress] = useState("Loading Python");
  const [error, setError] = useState<string | null>(null);
  const [refusal, setRefusal] = useState<{ reason: string; at: number } | null>(null);
  const [match, setMatch] = useState<MatchState | null>(null);
  const loaded = useRef(false);
  const waiting = useRef<Setup | null>(null); // a match asked for before Python was ready
  const expecting = useRef(false); // a match was asked for and not left since

  const send = useCallback((message: object) => worker.current?.postMessage(message), []);

  const begin = useCallback(
    (setup: Setup) => {
      setMatch(null);
      setError(null);
      setRefusal(null);
      expecting.current = true;
      if (!loaded.current) {
        waiting.current = setup;
        setPhase("starting");
        return;
      }
      setPhase("starting");
      send({ type: "start", ...setup, tickSeconds });
    },
    [send, tickSeconds],
  );

  useEffect(() => {
    const created = new Worker("/workers/play.js", { type: "module" });
    worker.current = created;
    created.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const message = event.data;
      if (message.type === "progress") setProgress(message.text);
      else if (message.type === "loaded") {
        loaded.current = true;
        if (waiting.current) {
          created.postMessage({ type: "start", ...waiting.current, tickSeconds });
          waiting.current = null;
        } else setPhase("ready");
      } else if (message.type === "started") {
        if (!expecting.current) return; // left while it was being set up; it stays paused
        setMatch(fromStart(message.start));
        setPhase("countdown");
      } else if (message.type === "tick") {
        setMatch((current) => (current ? withUpdate(current, message.update, message.pace) : current));
        if (message.update.outcome) setPhase("over");
        const refused = Object.values(message.update.refused)[0];
        if (refused) setRefusal({ reason: refused, at: performance.now() });
      } else if (message.type === "refused") setRefusal({ reason: message.reason, at: performance.now() });
      else if (message.type === "error") {
        setError(message.message);
        setPhase("failed");
      }
    };
    created.onerror = (event) => {
      setError(event.message || "The game could not run in this browser.");
      setPhase("failed");
    };
    created.postMessage({ type: "load", pyodideUrl: PYODIDE_URL, bundleUrl: `/play/${gameId}/`, gameId });
    return () => {
      created.terminate();
      worker.current = null;
      loaded.current = false;
    };
  }, [gameId, tickSeconds]);

  return {
    phase,
    progress,
    error,
    refusal,
    match,
    start: begin,
    go: useCallback(() => {
      send({ type: "resume" });
      setPhase("playing");
    }, [send]),
    pause: useCallback(() => {
      send({ type: "pause" });
      setPhase("paused");
    }, [send]),
    resume: useCallback(() => {
      send({ type: "resume" });
      setPhase("playing");
    }, [send]),
    leave: useCallback(() => {
      send({ type: "pause" });
      expecting.current = false;
      waiting.current = null;
      setMatch(null);
      setPhase(loaded.current ? "ready" : "loading");
    }, [send]),
    speed: useCallback((value: number) => send({ type: "speed", speed: value }), [send]),
    command: useCallback((seat: string, data: unknown) => send({ type: "command", seat, data }), [send]),
  };
}

function fromStart(start: MatchStart): MatchState {
  return {
    board: start.board,
    snapshot: start.snapshot,
    tick: start.tick,
    playedAt: performance.now(),
    people: start.people,
    thinking: {},
    errors: {},
    outcome: null,
    pace: 1,
  };
}

function withUpdate(state: MatchState, update: MatchUpdate, pace: number): MatchState {
  return {
    ...state,
    snapshot: applyDelta(state.snapshot, update.delta),
    tick: update.tick,
    playedAt: performance.now(),
    people: update.people,
    thinking: update.thinking_ms,
    errors: update.errors,
    outcome: update.outcome,
    pace,
  };
}
