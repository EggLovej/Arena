"use client";

// The play page's hold on its worker (public/workers/play.js), which runs the match in
// Python. Python starts loading as soon as the page opens, while the person picks a map
// and opponents, since that is what takes longest.

import { useCallback, useEffect, useRef, useState } from "react";

import { applyDelta } from "@/lib/frames.ts";
import { PYODIDE_URL, type MatchStart, type MatchUpdate, type Outcome, type WorkerMessage } from "@/lib/play.ts";
import type { Snapshot } from "@/lib/types.ts";

// "broken": Python could not load, so no match can start; "failed": a match stopped with an error.
export type Phase = "loading" | "broken" | "ready" | "starting" | "countdown" | "playing" | "paused" | "over" | "failed";

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
  const broken = useRef(false);
  const waiting = useRef<object | null>(null); // a match asked for before Python was ready
  // The match the page shows, numbered; messages about any other (one left, or replaced) are ignored.
  const current = useRef(0);

  const send = useCallback((message: object) => worker.current?.postMessage(message), []);

  const begin = useCallback(
    (setup: Setup) => {
      if (broken.current) return;
      setMatch(null);
      setError(null);
      setRefusal(null);
      setPhase("starting");
      const message = { type: "start", id: (current.current += 1), ...setup, tickSeconds };
      if (loaded.current) send(message);
      else waiting.current = message;
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
          created.postMessage(waiting.current);
          waiting.current = null;
        } else setPhase("ready");
      } else if (message.type === "error" && message.id === undefined) {
        breakDown(message.message);
      } else if (message.id !== current.current) {
        return; // about a match the person has left
      } else if (message.type === "started") {
        setMatch(fromStart(message.start));
        setPhase("countdown");
      } else if (message.type === "tick") {
        setMatch((state) => (state ? withUpdate(state, message.update, message.pace) : state));
        if (message.update.outcome) setPhase("over");
        const refused = Object.values(message.update.refused)[0];
        if (refused) setRefusal({ reason: refused, at: performance.now() });
      } else if (message.type === "refused") setRefusal({ reason: message.reason, at: performance.now() });
      else if (message.type === "error") {
        setError(message.message);
        setPhase("failed");
      }
    };
    // Python did not load, or the worker itself failed: no match can be played until the page is reloaded.
    const breakDown = (reason: string) => {
      broken.current = true;
      waiting.current = null;
      current.current += 1;
      setMatch(null);
      setError(reason);
      setPhase("broken");
    };
    created.onerror = (event) => breakDown(event.message || "The game could not run in this browser.");
    created.postMessage({ type: "load", pyodideUrl: PYODIDE_URL, bundleUrl: `/play/${gameId}/`, gameId });
    return () => {
      created.terminate();
      worker.current = null;
      loaded.current = false;
      broken.current = false;
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
      current.current += 1;
      waiting.current = null;
      setMatch(null);
      setPhase(broken.current ? "broken" : loaded.current ? "ready" : "loading");
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
