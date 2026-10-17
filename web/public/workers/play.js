// The play page's game, run in a background worker so the page stays smooth while bots
// think. It loads Python (Pyodide) and the game's code (public/play/<game>/python.json,
// written by scripts/bundle-python.mjs), plays the match with platform/arena/live.py,
// and keeps the clock: right after each tick the bots choose their next moves, then
// the worker waits until the tick is due and plays it with whatever the person did
// meanwhile. When bots think longer than a tick lasts, the game slows down to wait for them.
//
// Messages in: load, start, command, pause, resume, speed. Messages out: progress,
// loaded, started, tick, refused, error; those about a match carry the id the page gave
// it on start, so the page can tell a match it has left. web/src/lib/play.ts is the page's side.
// It is a module worker, because Pyodide no longer runs in classic ones.

let pyodide = null;
let session = null;
let game = null;
let files = "";
let round = 0; // which match is playing; a new one ends the last one's clock
let matchId = null; // the page's id for it
let paused = false;
let speed = 1;
let tickSeconds = 0.1;
let wake = null; // ends the playing match's wait early, on resume or when a new match starts

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === "load") await load(message);
    else if (message.type === "start") await start(message);
    else if (message.type === "command") {
      const refused = JSON.parse(session.command(message.seat, JSON.stringify(message.data)));
      if (refused) self.postMessage({ type: "refused", id: matchId, reason: refused });
    } else if (message.type === "pause") {
      paused = true;
    } else if (message.type === "resume") {
      paused = false;
      wake?.();
    } else if (message.type === "speed") {
      speed = message.speed;
    }
  } catch (error) {
    fail(error, message.type === "load" ? undefined : matchId);
  }
};

function progress(text) {
  self.postMessage({ type: "progress", text });
}

function fail(error, id) {
  round += 1;
  self.postMessage({ type: "error", id, message: String(error?.message ?? error) });
}

async function load({ pyodideUrl, bundleUrl, gameId }) {
  const started = performance.now();
  game = gameId;
  files = `${bundleUrl}files/`;
  progress("Loading Python");
  const { loadPyodide } = await import(`${pyodideUrl}pyodide.mjs`);
  const [python, bundle] = await Promise.all([
    loadPyodide({ indexURL: pyodideUrl }),
    fetch(`${bundleUrl}python.json`).then((response) => {
      if (!response.ok) throw new Error(`The game's code could not be loaded (${response.status}).`);
      return response.json();
    }),
  ]);
  pyodide = python;
  progress("Setting up the game");
  for (const [file, text] of Object.entries(bundle.files)) {
    const target = `/arena/${file}`;
    pyodide.FS.mkdirTree(target.slice(0, target.lastIndexOf("/")));
    pyodide.FS.writeFile(target, text);
  }
  pyodide.runPython('import os, sys\nos.environ["ARENA_ROOT"] = "/arena"\nsys.path.insert(0, "/arena/platform")');
  session = pyodide.pyimport("arena.live").Session(gameId);
  self.postMessage({ type: "loaded", seconds: (performance.now() - started) / 1000 });
}

async function start({ id, map, lineup, seed, tickSeconds: seconds }) {
  const mine = (round += 1);
  matchId = id;
  wake?.();
  tickSeconds = seconds;
  const needs = JSON.parse(session.needs(JSON.stringify(lineup)));
  if (needs.packages.length) {
    progress("Loading the bots' maths");
    await pyodide.loadPackage(needs.packages, { messageCallback: () => {} });
  }
  for (const [number, file] of needs.files.entries()) {
    progress(needs.files.length > 1 ? `Loading the bots' networks (${number + 1} of ${needs.files.length})` : "Loading the bot's network");
    const response = await fetch(files + file);
    if (!response.ok) throw new Error(`${file} could not be loaded (${response.status}).`);
    const target = `/arena/games/${game}/${file}`;
    pyodide.FS.mkdirTree(target.slice(0, target.lastIndexOf("/")));
    pyodide.FS.writeFile(target, new Uint8Array(await response.arrayBuffer()));
  }
  progress("Setting out the board");
  const first = JSON.parse(session.start(map, JSON.stringify(lineup), seed));
  // The bots' first moves can take a moment (loading a network); they are made before the clock starts.
  session.think();
  if (mine !== round) return;
  paused = true;
  self.postMessage({ type: "started", id, start: first });
  play(mine, id).catch((error) => fail(error, id));
}

/** Waits until woken, or for `ms` if given. Only clears `wake` while it is still this wait's, so a
 * match that has been replaced cannot take the new match's. */
async function rest(ms) {
  let resolve;
  const done = new Promise((settle) => (resolve = settle));
  wake = resolve;
  const timer = ms === undefined ? null : setTimeout(resolve, ms);
  await done;
  clearTimeout(timer);
  if (wake === resolve) wake = null;
}

async function play(mine, id) {
  let due = performance.now();
  let pace = 1;
  let last = null;
  while (mine === round) {
    if (paused) {
      await rest();
      due = performance.now() + (tickSeconds * 1000) / speed;
      last = null;
      continue;
    }
    // Always yield, even when late, so the person's clicks get through.
    await rest(Math.max(0, due - performance.now()));
    if (mine !== round || paused) continue;
    const now = performance.now();
    const length = (tickSeconds * 1000) / speed;
    if (last !== null) pace = 0.9 * pace + 0.1 * Math.min(1, length / Math.max(1, now - last));
    last = now;
    const update = JSON.parse(session.step());
    self.postMessage({ type: "tick", id, update, pace });
    if (update.outcome) return;
    // On time, ticks keep a steady beat. A late tick (the bots thought long) is not made up
    // in a rush: the next comes a whole tick after it, so the person still has time to act.
    due = now - due > length / 4 ? now + length : due + length;
    session.think();
  }
}
