// Epoch's lane in the browser: the backdrop and sprites in public/games/epoch (see
// the README there), the two bases with their turrets, the armies marching in their
// three files, shots in flight, and above each base its age, hit points and the
// units waiting to leave it.

import type { BoardProps } from "../types.ts";

type Role = "melee" | "ranged" | "heavy";

export type EpochBoardData = {
  name: string;
  length: number; // the lane, bases included; the board is this wide
  base_length: number;
  queue: number; // units that can wait at a base
  places: number[]; // what opening each turret place costs
  ages: { id: string; name: string; xp: number; price: number; base_hp: number; units: string[]; turret: string }[];
  units: Record<string, { name: string; role: Role; price: number; train: number; hp: number; reach: number; speed: number; length: number }>;
  turrets: Record<string, { name: string; price: number; reach: number }>;
};

export type EpochTheme = { sky: string; ink: string; panel: string; players: Record<string, string> };

// [age, base hit points, gold, XP, open turret places]
export type SideState = [number, number, number, number, number];
// [seat, unit id, file, position, tick it was there, walking (1 or 0), hit points, tick it last struck]
type UnitState = [string, string, number, number, number, number, number, number];
// [turret id, tick it last fired]
type TurretState = [string, number];
// [tick the first unit began training, unit ids...]
type QueueState = [number, ...string[]];

export const HEIGHT = 320;
export const SPRITES = "/games/epoch/sprites.svg";
export const SIZES: Record<Role, [number, number]> = { melee: [48, 48], ranged: [48, 48], heavy: [80, 64] };
// The ground under each file: the first a unit takes is in the middle, the second behind it, the third in front.
const FEET = [HEIGHT - 26, HEIGHT - 38, HEIGHT - 14];
const BASE_FOOT = HEIGHT - 40;
// Where shots leave each shooter, from the bottom centre of its sprite facing right, and what they look like.
const SHOTS: Record<string, [number, number, "rock" | "boulder" | "bolt" | "laser"]> = {
  slinger: [20, -28, "rock"],
  rock_thrower: [8, -32, "rock"],
  crossbowman: [16, -33, "bolt"],
  arrow_tower: [17, -22, "bolt"],
  catapult: [6, -60, "boulder"],
  laser_trooper: [17, -31, "laser"],
  mech: [31, -40, "laser"],
  laser_turret: [19, -18, "laser"],
};
const STEP = 0.2; // seconds each walking frame, and a strike, is shown
const FLIGHT = 0.25; // seconds a shot is drawn flying

/** A sprite with its bottom centre at (x, y), facing right or, flipped, left, in a seat's colour. */
export function Sprite({ id, x, y, size: [w, h], flip, color }: { id: string; x: number; y: number; size: [number, number]; flip: boolean; color: string }) {
  return (
    <g transform={`translate(${x} ${y})${flip ? " scale(-1 1)" : ""}`} style={{ color }}>
      <use href={`${SPRITES}#${id}`} x={-w / 2} y={-h} width={w} height={h} />
    </g>
  );
}

export function EpochBoard({ board: rawBoard, snapshot, next, tick, tickSeconds, theme: rawTheme, seats, compact = false }: BoardProps) {
  const board = rawBoard as EpochBoardData;
  const theme = rawTheme as EpochTheme;
  const { length, base_length: baseLength } = board;
  const sides = (snapshot.sides ?? {}) as Record<string, SideState>;
  const units = Object.entries((snapshot.units ?? {}) as Record<string, UnitState>);
  const nextUnits = (next?.units ?? {}) as Record<string, UnitState>;
  const turrets = (snapshot.turrets ?? {}) as Record<string, TurretState>;
  const queues = (snapshot.queues ?? {}) as Record<string, QueueState>;
  const backdrop = Math.max(HEIGHT, length * 0.3);
  const whole = Math.floor(tick);
  const color = (seat: string) => theme.players[seat] ?? seats.find((entry) => entry.id === seat)?.color ?? theme.ink;
  // Each side measures from its own end of the lane; the first seat's end is on the left.
  const flipped = (seat: string) => seat !== seats[0]?.id;
  const onScreen = (seat: string, x: number) => (flipped(seat) ? length - x : x);
  const ago = (when: number) => (tick - when) * tickSeconds; // seconds since a tick, for strikes and shots

  // Where a unit is: a walking unit moves on from where it was, and between two ticks it moves smoothly.
  const at = (state: UnitState, when: number) => (state[5] ? state[3] + board.units[state[1]].speed * (when - state[4]) : state[3]);
  const placed = units.map(([key, state]) => {
    const after = nextUnits[key];
    const now = at(state, whole);
    const x = after ? now + (at(after, whole + 1) - now) * (tick - whole) : at(state, next ? whole : tick);
    return { key, state, x, moving: state[5] === 1 || (after !== undefined && at(after, whole + 1) !== now) };
  });
  placed.sort((a, b) => FEET[a.state[2]] - FEET[b.state[2]]); // the back file first

  // How far from each side's end the enemy is nearest: its front unit's near edge, else its base.
  const nearest: Record<string, number> = {};
  for (const seat of Object.keys(sides)) nearest[seat] = length - baseLength;
  for (const { state, x } of placed) {
    for (const seat of Object.keys(sides)) {
      if (seat !== state[0]) nearest[seat] = Math.min(nearest[seat], length - x - board.units[state[1]].length / 2);
    }
  }

  /** A shot from (x, y) in a seat's own measure toward the enemy at distance `target`, `seconds` after it left. */
  const shot = (key: string, seat: string, kind: string, x: number, y: number, target: number, seconds: number) => {
    const [dx, dy, look] = SHOTS[kind];
    const progress = seconds / FLIGHT;
    const x0 = x + dx;
    const y0 = y + dy;
    const along = (p: number) => onScreen(seat, x0 + (target - x0) * p);
    const lift = { rock: 24, boulder: 50, bolt: 0, laser: 0 }[look] * 4 * progress * (1 - progress);
    const y1 = y0 + (FEET[0] - 22 - y0) * progress - lift;
    if (look === "laser") {
      const tail = Math.max(0, progress - 0.35);
      const ty = y0 + (FEET[0] - 22 - y0) * tail;
      return (
        <g key={key} strokeLinecap="round">
          <line x1={along(tail)} y1={ty} x2={along(progress)} y2={y1} stroke={color(seat)} strokeWidth={4} opacity={0.6} />
          <line x1={along(tail)} y1={ty} x2={along(progress)} y2={y1} stroke="#fff" strokeWidth={1.5} />
        </g>
      );
    }
    if (look === "bolt") {
      const back = onScreen(seat, x0 + (target - x0) * progress - 12);
      return <line key={key} x1={back} y1={y1} x2={along(progress)} y2={y1} stroke={theme.ink} strokeWidth={2.5} strokeLinecap="round" />;
    }
    return <circle key={key} cx={along(progress)} cy={y1} r={look === "boulder" ? 6 : 3.5} fill="#a7a196" stroke={theme.ink} strokeWidth={1.5} />;
  };

  return (
    <svg
      viewBox={`0 0 ${length} ${HEIGHT}`}
      preserveAspectRatio="xMidYMax meet" // any room to spare is more sky
      role="img"
      aria-label={`${board.name} lane`}
      style={{ display: "block", width: "100%", height: compact ? "100%" : "auto", background: theme.sky }}
    >
      {/* The backdrop is 1000 by 300; as tall as the board's width needs, a long lane's sky reaches above the top. */}
      <image href="/games/epoch/lane.svg" y={HEIGHT - backdrop} width={length} height={backdrop} preserveAspectRatio="xMidYMax slice" />

      {Object.entries(sides).map(([seat, [age, hp, , , places]]) => {
        const flip = flipped(seat);
        const max = board.ages[age].base_hp;
        const queue = queues[seat] ?? [0];
        const corner = flip ? length - 116 : 4; // above the base
        return (
          <g key={seat}>
            <Sprite id={`base_${board.ages[age].id}`} x={onScreen(seat, baseLength / 2)} y={BASE_FOOT} size={[120, 150]} flip={flip} color={color(seat)} />
            {Array.from({ length: places }, (_, slot) => {
              const turret = turrets[`${seat}-${slot}`];
              const [x, y] = [baseLength / 2 + 40, BASE_FOOT - 40 * (slot + 1)];
              if (!turret) return null;
              const fired = ago(turret[1]);
              return (
                <g key={slot}>
                  <Sprite id={turret[0]} x={onScreen(seat, x)} y={y} size={[40, 40]} flip={flip} color={color(seat)} />
                  {!compact && turret[1] >= 0 && fired >= 0 && fired < FLIGHT && shot(`shot-${seat}-${slot}`, seat, turret[0], x, y, nearest[seat] + 12, fired)}
                </g>
              );
            })}
            {!compact && (
              <g transform={`translate(${corner} 62)`}>
                <g fontSize={13} fontWeight={700} fill={theme.ink} stroke={theme.panel} strokeWidth={3} paintOrder="stroke">
                  <text y={12}>{board.ages[age].name}</text>
                  <text x={112} y={12} textAnchor="end">
                    {Math.max(0, hp)}
                  </text>
                </g>
                <rect y={18} width={112} height={8} rx={4} fill={theme.panel} stroke={theme.ink} strokeWidth={1} />
                <rect y={18} width={(112 * Math.max(0, hp)) / max} height={8} rx={4} fill={color(seat)} />
                {(queue.slice(1) as string[]).map((id, index) => (
                  <Sprite key={index} id={`${id}-walk-1`} x={index * 24 + 11} y={56} size={[22, 22]} flip={flip} color={color(seat)} />
                ))}
                {queue.length > 1 && (
                  <rect y={59} width={22 * Math.min(1, Math.max(0, tick - queue[0]) / board.units[queue[1] as string].train)} height={3} rx={1.5} fill={color(seat)} />
                )}
              </g>
            )}
          </g>
        );
      })}

      {placed.map(({ key, state: [seat, kind, file, , , , hp, struck], x, moving }) => {
        const unit = board.units[kind];
        const [w, h] = SIZES[unit.role];
        const since = struck >= 0 ? ago(struck) : -1; // seconds since it last struck
        const striking = since >= 0 && since < STEP;
        const frame = striking ? "attack" : moving && Math.floor((tick * tickSeconds) / STEP + Number(key)) % 2 ? "walk-2" : "walk-1";
        const y = FEET[file];
        return (
          <g key={key}>
            <Sprite id={`${kind}-${frame}`} x={onScreen(seat, x)} y={y} size={[w, h]} flip={flipped(seat)} color={color(seat)} />
            {!compact && hp < unit.hp && (
              <g transform={`translate(${onScreen(seat, x) - 12} ${y - h - 2})`}>
                <rect width={24} height={4} rx={2} fill={theme.panel} stroke={theme.ink} strokeWidth={0.75} />
                <rect width={(24 * hp) / unit.hp} height={4} rx={2} fill={color(seat)} />
              </g>
            )}
            {!compact && kind in SHOTS && since >= 0 && since < FLIGHT && shot(`shot-${key}`, seat, kind, x, y, nearest[seat] + 10, since)}
          </g>
        );
      })}
    </svg>
  );
}
