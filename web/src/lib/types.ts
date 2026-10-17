// The shapes of the JSON files in public/data, written by platform/arena/publish.py.

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
  leaderboard: boolean;
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

export type GameSummary = {
  id: string;
  name: string;
  tagline: string;
  seats: string[];
  maps: number;
  bots: number;
};

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
  average_place: number;
  errors: number;
  refused: number;
  think_ms_mean: number;
  think_ms_max: number;
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
  game: string;
  updated: string;
  inputs: string;
  fingerprint: string;
  settings: { bots: string[] | null; maps: string[] | null; rounds: number };
  matches: number;
  replays: number;
  cpu_seconds: number;
  wall_seconds: number;
  tables: Record<string, Table>;
};

export type MatchRow = {
  id: string;
  map: string;
  lineup: Record<string, string>;
  ranking: string[][];
  reason: string;
  ticks: number;
  replay: boolean;
};

export type Frames = {
  keyframe_every: number;
  count: number;
  keyframes: Record<string, Snapshot>;
  deltas: Delta[];
};

export type SeatReport = {
  bot: string;
  moves: number;
  errors: number;
  refused: number;
  think_ms_mean: number;
  think_ms_max: number;
  first_error: string;
};

export type Replay = {
  format: "arena-replay";
  version: number;
  game: string;
  fingerprint: string;
  match_id: string;
  map: string;
  seed: number;
  tick_seconds: number;
  max_ticks: number;
  seats: { seat: string; bot: string }[];
  board: unknown;
  frames: Frames;
  actions: [number, string, unknown][];
  hashes: Record<string, string>;
  outcome: { ranking: string[][]; reason: string; ticks: number };
  reports: Record<string, SeatReport>;
};
