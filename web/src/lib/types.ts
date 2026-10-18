// The shapes of the JSON files in public/data, written by platform/arena/publish.py:
// the fields the site reads.

export type Seat = { id: string; name: string; color: string };

export type Snapshot = Record<string, Record<string, unknown>>;
export type Delta = Record<string, Record<string, unknown>>;

export type MapEntry = {
  id: string;
  name: string;
  seats: string[];
  description: string;
  max_ticks?: number | null; // the map's own time limit, when it differs from the game's
  group?: string; // a heading to list the map under; "" for the main list
  board: unknown;
  start: Snapshot;
};

export type BotEntry = {
  id: string;
  name: string;
  description: string;
  kind: string;
  seat_counts: number[] | null;
  sits_out?: string[]; // map groups the bot leaves out of tournaments
};

export type GameData = {
  id: string;
  name: string;
  tagline: string;
  seats: Seat[];
  tick_seconds: number;
  max_ticks: number;
  rating_anchor: [string, number] | null;
  rules: string;
  theme: unknown;
  maps: MapEntry[];
  bots: BotEntry[];
};

/** An entry in games.json; the rest is in each game's own game.json. */
export type GameSummary = { id: string };

export type Standing = {
  bot: string;
  name: string;
  kind: string;
  rating: number;
  interval: number;
  games: number;
  wins: number;
  draws: number;
  losses: number;
  score: number;
  errors: number;
  refused: number;
  think_ms_mean: number;
  by_map: Record<string, { score: number; games: number }>;
};

export type Table = {
  players: number;
  label: string;
  matches: number;
  maps: string[];
  entries: Standing[];
  head_to_head: Record<string, Record<string, { points: number; games: number }>>;
};

export type Leaderboard = {
  updated: string;
  matches: number;
  tables: Record<string, Table>;
};

export type MatchRow = {
  id: string;
  map: string;
  lineup: Record<string, string>;
  ranking: string[][];
  ticks: number;
  replay: boolean;
};

export type Frames = {
  count: number;
  keyframes: Record<string, Snapshot>;
  deltas: Delta[];
};

/** How a match ended: the seats from first place down (seats that tie share a group), why, and after how many ticks. */
export type Outcome = { ranking: string[][]; reason: string; ticks: number };

export type Replay = {
  tick_seconds: number;
  seats: { seat: string; bot: string }[];
  board: unknown;
  frames: Frames;
  actions: [number, string, unknown][];
  outcome: Outcome;
};
