// Epoch's lane in the browser: the backdrop and sprites in public/games/epoch (see
// the README there), the two bases with their turrets, the armies marching in single
// file, shots in flight, the specials, and above each base its age, hit points and
// the units waiting to leave it.

import type { BoardProps } from "../types.ts";

type Role = "melee" | "ranged" | "heavy" | "super";

export type EpochBoardData = {
  name: string;
  length: number; // the lane, bases included; the board is this wide
  base_length: number;
  queue: number; // units that can wait at a base
  places: number[]; // what opening each turret place costs
  special_cooldown: number; // ticks between a side's specials
  ages: {
    id: string;
    name: string;
    xp: number;
    base_hp: number;
    units: string[];
    turrets: string[]; // cheapest first
    special: { id: string; name: string; heals: boolean };
  }[];
  units: Record<string, { name: string; role: Role; price: number; train: number; hp: number; reach: number; speed: number; length: number }>;
  turrets: Record<string, { name: string; price: number; reach: number }>;
};

export type EpochTheme = { sky: string; ink: string; panel: string; players: Record<string, string> };

// [age, base hit points, gold, XP, open turret places, tick its special last struck]
export type SideState = [number, number, number, number, number, number];
// [seat, unit id, position, tick it was there, walking (1 or 0), hit points, tick it last struck]
type UnitState = [string, string, number, number, number, number, number];
// [turret id, tick it last fired]
type TurretState = [string, number];
// [tick the first unit began training, unit ids...]
type QueueState = [number, ...string[]];

export const HEIGHT = 320;
export const SPRITES = "/games/epoch/sprites.svg";
export const SIZES: Record<Role, [number, number]> = { melee: [48, 48], ranged: [48, 48], heavy: [80, 64], super: [64, 64] };
const FOOT = HEIGHT - 26; // the ground the army marches on
const BASE_FOOT = HEIGHT - 40;
const PANEL = 150; // the width of the age, hit points and queue shown beside each base
const LEDGE = 36; // turret ledges are this far apart on a base, the first this far up, 40 in front of its middle
// Where shots leave each shooter, from the bottom centre of its sprite facing right, and what they look like.
type Look = "rock" | "boulder" | "bolt" | "laser";
const SHOTS: Record<string, [number, number, Look]> = {
  slinger: [20, -28, "rock"],
  crossbowman: [16, -33, "bolt"],
  catapult: [6, -60, "boulder"],
  musketeer: [18, -31, "bolt"],
  cannon: [33, -24, "boulder"],
  gunner: [19, -21, "bolt"],
  tank: [33, -28, "boulder"],
  laser_trooper: [17, -31, "laser"],
  mech: [31, -40, "laser"],
  super_soldier: [26, -38, "laser"],
  rock_thrower: [8, -32, "rock"],
  sling_nest: [-2, -28, "rock"],
  boulder_catapult: [2, -30, "boulder"],
  arrow_tower: [16, -12, "bolt"],
  ballista: [17, -19, "bolt"],
  oil_cauldron: [11, -16, "boulder"],
  small_cannon: [13, -15, "boulder"],
  large_cannon: [18, -19, "boulder"],
  mortar: [8, -20, "boulder"],
  machine_gun: [19, -15, "bolt"],
  rocket_launcher: [14, -21, "bolt"],
  twin_gun: [18, -17, "bolt"],
  laser_turret: [19, -18, "laser"],
  plasma_cannon: [12, -15, "laser"],
  ion_ray: [12, -24, "laser"],
};
const STEP = 0.2; // seconds each walking frame, and a strike, is shown
const FLIGHT = 0.25; // seconds a shot is drawn flying
const STRIKE = 0.8; // seconds a special is drawn

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
  const at = (state: UnitState, when: number) => (state[4] ? state[2] + board.units[state[1]].speed * (when - state[3]) : state[2]);
  const placed = units.map(([key, state]) => {
    const after = nextUnits[key];
    const now = at(state, whole);
    const x = after ? now + (at(after, whole + 1) - now) * (tick - whole) : at(state, next ? whole : tick);
    return { key, state, x, moving: state[4] === 1 || (after !== undefined && at(after, whole + 1) !== now) };
  });

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
    const y1 = y0 + (FOOT - 22 - y0) * progress - lift;
    if (look === "laser") {
      const tail = Math.max(0, progress - 0.35);
      const ty = y0 + (FOOT - 22 - y0) * tail;
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

  /** A seat's special, `seconds` after it struck: rocks, arrows, bombs or beams falling on the enemy's half
   * of the lane (its units there were hit at once), or sparkles rising from the seat's own units if it heals. */
  const special = (seat: string, id: string, seconds: number) => {
    const progress = seconds / STRIKE;
    if (id === "healing") {
      return placed
        .filter(({ state }) => state[0] === seat)
        .map(({ key, x }) => (
          <circle key={`heal-${key}`} cx={onScreen(seat, x)} cy={FOOT - 20 - 40 * progress} r={5} fill="#f6e27a" stroke={theme.ink} strokeWidth={1} opacity={1 - progress} />
        ));
    }
    const from = length / 2;
    const to = length - baseLength;
    return Array.from({ length: 12 }, (_, index) => {
      // Spread along the enemy's half, each a little later than the last, so they rain rather than drop as one.
      const x = onScreen(seat, from + ((to - from) * ((index * 7) % 12 + 0.5)) / 12);
      const fall = Math.min(1, Math.max(0, progress * 1.6 - (index % 4) * 0.15));
      const y = -20 + (FOOT + 20) * fall;
      const key = `${seat}-special-${index}`;
      if (fall <= 0 || fall >= 1) return null;
      if (id === "orbital_laser") {
        return (
          <g key={key} strokeLinecap="round">
            <line x1={x} y1={0} x2={x} y2={y} stroke={color(seat)} strokeWidth={6} opacity={0.6} />
            <line x1={x} y1={0} x2={x} y2={y} stroke="#fff" strokeWidth={2} />
          </g>
        );
      }
      if (id === "arrow_storm") {
        return <line key={key} x1={x - 6} y1={y - 14} x2={x} y2={y} stroke={theme.ink} strokeWidth={2.5} strokeLinecap="round" />;
      }
      const bomb = id === "air_strike";
      return <ellipse key={key} cx={x} cy={y} rx={bomb ? 4 : 6} ry={bomb ? 8 : 6} fill={bomb ? "#4c5662" : "#a7a196"} stroke={theme.ink} strokeWidth={1.5} />;
    });
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
        const corner = flip ? length - baseLength - 4 - PANEL : baseLength + 4; // in the sky just beyond the base, clear of its turrets
        return (
          <g key={seat}>
            <Sprite id={`base_${board.ages[age].id}`} x={onScreen(seat, baseLength / 2)} y={BASE_FOOT} size={[120, 150]} flip={flip} color={color(seat)} />
            {Array.from({ length: places }, (_, slot) => {
              const turret = turrets[`${seat}-${slot}`];
              const [x, y] = [baseLength / 2 + 40, BASE_FOOT - LEDGE * (slot + 1)];
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
              <g transform={`translate(${corner} 76)`}>
                <g fontSize={13} fontWeight={700} fill={theme.ink} stroke={theme.panel} strokeWidth={3} paintOrder="stroke">
                  <text y={12}>{board.ages[age].name}</text>
                  <text x={PANEL} y={12} textAnchor="end">
                    {Math.max(0, hp)}
                  </text>
                </g>
                <rect y={18} width={PANEL} height={8} rx={4} fill={theme.panel} stroke={theme.ink} strokeWidth={1} />
                <rect y={18} width={(PANEL * Math.max(0, hp)) / max} height={8} rx={4} fill={color(seat)} />
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

      {placed.map(({ key, state: [seat, kind, , , , hp, struck], x, moving }) => {
        const unit = board.units[kind];
        const [w, h] = SIZES[unit.role];
        const since = struck >= 0 ? ago(struck) : -1; // seconds since it last struck
        const striking = since >= 0 && since < STEP;
        const frame = striking ? "attack" : moving && Math.floor((tick * tickSeconds) / STEP + Number(key)) % 2 ? "walk-2" : "walk-1";
        return (
          <g key={key}>
            <Sprite id={`${kind}-${frame}`} x={onScreen(seat, x)} y={FOOT} size={[w, h]} flip={flipped(seat)} color={color(seat)} />
            {!compact && hp < unit.hp && (
              <g transform={`translate(${onScreen(seat, x) - 12} ${FOOT - h - 2})`}>
                <rect width={24} height={4} rx={2} fill={theme.panel} stroke={theme.ink} strokeWidth={0.75} />
                <rect width={(24 * hp) / unit.hp} height={4} rx={2} fill={color(seat)} />
              </g>
            )}
            {!compact && kind in SHOTS && since >= 0 && since < FLIGHT && shot(`shot-${key}`, seat, kind, x, FOOT, nearest[seat] + 10, since)}
          </g>
        );
      })}

      {!compact &&
        Object.entries(sides).map(([seat, [age, , , , , struck]]) => {
          const seconds = ago(struck);
          return seconds >= 0 && seconds < STRIKE ? <g key={`special-${seat}`}>{special(seat, board.ages[age].special.id, seconds)}</g> : null;
        })}
    </svg>
  );
}
