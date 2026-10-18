// Robot League's style (display/styles/robot_league.py): a worn concrete court in
// a steel frame with safety paint and lamps, steel walls on soft shadows, and
// tracked machines. A snapshot doesn't say which way a tank drives, so its hull
// faces where it aims. The concrete and steel are the game's own pictures
// (src/ricochet/data/themes/robot_league), the concrete softened as the game
// does it, both made small for the web.

import type { Style } from "./RicochetBoard.tsx";

const CONCRETE = "/games/ricochet/concrete.webp";
const STEEL = "/games/ricochet/steel.webp";

// The materials' colours, named as in robot_league.py. The safety paint is the theme's accent.
const JOINT_SHADOW = "#626667";
const JOINT_HIGHLIGHT = "#a9abaa";
const EDGE_SHADE = "#0d131b";
const FRAME = "#1e252b";
const FRAME_GROOVE = "#090e12";
const FRAME_EDGE = "#8a979d";
const RAIL_EDGE = "#707c84";
const SIDE_RAIL = "#4e585e";
const CORNER_ARMOR = "#3e484f";
const CORNER_EDGE = "#a7b1b4";
const HAZARD_STRIPE = "#39382d";
const LAMP_HOUSING = "#0b0f13";
const LAMP_RIM = "#9e8f6f";
const LAMP_LENS = "#ffd285";
const LAMP_GLOW = "#ffce7a";
const BOLT_SHADOW = "#12171c";
const BOLT_RIM = "#9ba6ab";
const BOLT_FACE = "#353e45";
const SHADOW = "#070c11";
const WALL_BODY = "#191e22";
const WALL_EDGE = "#a5b1b6";
const WALL_BEVEL = "#636f75";
const TRACK = "#15191b";
const TREAD = "#30373b";
const BARREL_SHADOW = "#121b20";
const BARREL = "#8b9ba4";
const HUB_SHADOW = "#131d23";
const HUB_RIM = "#a4b3b8";
const HUB_CORE = "#273a44";
const HUB_CAP = "#789faf";
const WRECK = "#4b4d4c";

const DESIGN_RADIUS = 18; // the machines are drawn for this tank radius and scale with the rules'

type Rect = [number, number, number, number]; // x, y, width, height

/** The outline of a rectangle with its corners cut off at 45 degrees, from the top edge round. */
function chamfer([x, y, w, h]: Rect, cut: number): [number, number][] {
  return [[x + cut, y], [x + w - cut, y], [x + w, y + cut], [x + w, y + h - cut], [x + w - cut, y + h], [x + cut, y + h], [x, y + h - cut], [x, y + cut]];
}

function bolt(x: number, y: number, r: number) {
  return (
    <g key={`${x},${y}`}>
      <circle cx={x} cy={y + 1} r={r + 1} fill={BOLT_SHADOW} />
      <circle cx={x} cy={y} r={r} fill={BOLT_RIM} />
      <circle cx={x} cy={y} r={Math.max(1, r - 1)} fill={BOLT_FACE} />
    </g>
  );
}

/** A lamp in the frame at (x, y), shining in direction (dx, dy). */
function lamp(x: number, y: number, dx: number, dy: number, glow: string) {
  const [w, h] = dy ? [29, 10] : [10, 29];
  return (
    <g key={`${x},${y}`}>
      <circle cx={x + dx * 24} cy={y + dy * 24} r={70} fill={glow} />
      <rect x={x - w / 2 - 2.5} y={y - h / 2 - 2} width={w + 5} height={h + 4} rx={2} fill={LAMP_HOUSING} />
      <rect x={x - w / 2} y={y - h / 2} width={w} height={h} rx={2} fill={LAMP_LENS} stroke={LAMP_RIM} />
    </g>
  );
}

function machine(color: string, radius: number, aim: number) {
  return (
    <g transform={`scale(${radius / DESIGN_RADIUS})`}>
      <ellipse cy={5} rx={20.5} ry={18} fill={SHADOW} opacity={0.35} />
      <g transform={`rotate(${aim})`}>
        {/* The hull, nose along x: rubber tracks either side of the armour, vents at the back. */}
        {[-13, 13].map((y) => (
          <g key={y}>
            <rect x={-17} y={y - 4} width={34} height={8} rx={2} fill={TRACK} />
            <line x1={-16} y1={y} x2={16} y2={y} stroke={TREAD} strokeWidth={6} strokeDasharray="1.6 1.8" />
          </g>
        ))}
        <rect x={-15} y={-9} width={30} height={18} rx={2} fill={color} stroke={SHADOW} strokeOpacity={0.5} />
        <rect x={-14} y={-5} width={3} height={10} fill={SHADOW} opacity={0.45} />
        {/* The gun and the turret's hub. */}
        <rect y={-2.4} width={25} height={4.8} fill={BARREL_SHADOW} />
        <rect x={3} y={-1.5} width={20} height={3} fill={BARREL} />
        <rect x={6} y={-2} width={8} height={4} fill={color} />
        {(
          [
            [8, HUB_SHADOW],
            [7, HUB_RIM],
            [6, color],
            [4, HUB_CORE],
            [2, HUB_CAP],
          ] as const
        ).map(([r, fill]) => (
          <circle key={r} r={r} fill={fill} />
        ))}
      </g>
    </g>
  );
}

export const robotLeague: Style = {
  arena({ width, height, walls }, theme, compact, id) {
    const paint = `url(#${id}-hazard)`;
    // Safety paint round the edges, broken where the lamps are.
    const strips: Rect[] = [
      ...[37, width - 167].flatMap((x) => [25, height - 39].map((y): Rect => [x, y, 130, 14])),
      ...[61, height - 196].flatMap((y) => [25, width - 39].map((x): Rect => [x, y, 14, 135])),
      ...[25, width - 35].map((x): Rect => [x, height / 2 - 66, 10, 132]),
    ];
    const lamps = [
      ...[1, 3, 5, 7].flatMap((eighth) => [lamp((width * eighth) / 8, 13, 0, 1, `url(#${id}-lamp)`), lamp((width * eighth) / 8, height - 14, 0, -1, `url(#${id}-lamp)`)]),
      ...[1, 3].flatMap((quarter) => [lamp(13, (height * quarter) / 4, 1, 0, `url(#${id}-lamp)`), lamp(width - 14, (height * quarter) / 4, -1, 0, `url(#${id}-lamp)`)]),
    ];
    return (
      <>
        <defs>
          <pattern id={`${id}-joints`} width={160} height={128} patternUnits="userSpaceOnUse">
            <path d="M 0 0.5 H 160 M 0.5 0 V 128" stroke={JOINT_SHADOW} />
            <path d="M 1 1.5 H 160" stroke={JOINT_HIGHLIGHT} />
          </pattern>
          <pattern id={`${id}-hazard`} width={16 / Math.SQRT2} height={16} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width={16} height={16} fill={theme.accent} />
            <rect width={7 / Math.SQRT2} height={16} fill={HAZARD_STRIPE} />
          </pattern>
          {/* Stretched over each shape it fills, as the game scales it to each wall. */}
          <pattern id={`${id}-steel`} width={1} height={1} patternContentUnits="objectBoundingBox">
            <image href={STEEL} width={1} height={1} preserveAspectRatio="none" />
          </pattern>
          <radialGradient id={`${id}-shade`} r="71%">
            <stop offset="55%" stopColor={EDGE_SHADE} stopOpacity={0} />
            <stop offset="100%" stopColor={EDGE_SHADE} stopOpacity={0.4} />
          </radialGradient>
          <radialGradient id={`${id}-lamp`}>
            <stop offset="0%" stopColor={LAMP_GLOW} stopOpacity={0.3} />
            <stop offset="100%" stopColor={LAMP_GLOW} stopOpacity={0} />
          </radialGradient>
          <clipPath id={`${id}-court`}>
            <rect width={width} height={height} />
          </clipPath>
        </defs>

        <image href={CONCRETE} width={width} height={height} preserveAspectRatio="none" />
        <rect width={width} height={height} fill={`url(#${id}-joints)`} />
        {!compact && <rect width={width} height={height} fill={`url(#${id}-shade)`} />}

        {/* The frame: the arena's edge, which shots bounce off like any wall. */}
        <rect x={7.5} y={7.5} width={width - 15} height={height - 15} fill="none" stroke={FRAME} strokeWidth={15} />
        {!compact && (
          // The lamps' light and the last rail stop at the arena's edge, where a wider frame shows the background.
          <g clipPath={`url(#${id}-court)`}>
            <rect x={15.5} y={15.5} width={width - 31} height={height - 31} fill="none" stroke={FRAME_GROOVE} strokeWidth={3} />
            <rect x={1.5} y={1.5} width={width - 3} height={height - 3} fill="none" stroke={FRAME_EDGE} />
            {Array.from({ length: Math.ceil((width - 40) / 80) }, (_, index) =>
              [2, height - 13].map((y) => <rect key={`${index},${y}`} x={20 + index * 80} y={y} width={73} height={10} fill={`url(#${id}-steel)`} stroke={RAIL_EDGE} />),
            )}
            {Array.from({ length: Math.ceil((height - 47) / 60) }, (_, index) =>
              [3, width - 12].map((x) => <rect key={`${index},${x}`} x={x} y={23 + index * 60} width={8} height={54} fill="none" stroke={SIDE_RAIL} />),
            )}
            {[width / 3, (width * 2) / 3].flatMap((x) => [2, height - 13].map((y) => <rect key={`${x},${y}`} x={x - 12} y={y} width={24} height={11} fill={theme.accent} />))}
            {strips.map(([x, y, w, h]) => (
              <rect key={`${x},${y}`} x={x} y={y} width={w} height={h} fill={paint} />
            ))}
            {[0, width].flatMap((x, i) =>
              [0, height].map((y, j) => {
                const [sx, sy] = [i ? -1 : 1, j ? -1 : 1];
                const armour = [[x, y], [x + sx * 32, y], [x + sx * 13, y + sy * 26], [x, y + sy * 32]];
                return (
                  <g key={`${x},${y}`}>
                    <polygon points={armour.join(" ")} fill={CORNER_ARMOR} />
                    <polyline points={armour.slice(1).join(" ")} fill="none" stroke={CORNER_EDGE} />
                    <line x1={x + sx * 9} y1={y + sy * 25} x2={x + sx * 25} y2={y + sy * 9} stroke={theme.accent} strokeWidth={5} />
                  </g>
                );
              }),
            )}
            {lamps}
          </g>
        )}

        {walls.map((wall, index) => {
          const [x, y, w, h] = wall;
          const top: Rect = [x + 6, y + 5, w - 14, h - 14];
          const outline = chamfer(wall, Math.min(7, w / 5, h / 5));
          return (
            <g key={index}>
              {!compact &&
                [10, 6, 3].map((spread) => (
                  <rect key={spread} x={x + 5 - spread} y={y + 12 - spread} width={w + 2 * spread} height={h + 2 * spread} rx={5} fill={SHADOW} opacity={0.14} />
                ))}
              <polygon points={outline.join(" ")} fill={WALL_BODY} />
              {top[2] > 0 && top[3] > 0 && <polygon points={chamfer(top, 4).join(" ")} fill={`url(#${id}-steel)`} stroke={WALL_BEVEL} />}
              <polyline points={outline.slice(0, 3).join(" ")} fill="none" stroke={WALL_EDGE} />
              {!compact && h > 45 && [x + 1, x + w - 6].map((left) => <rect key={left} x={left} y={y + 16} width={5} height={h - 32} fill={paint} />)}
              {!compact &&
                top[2] > 0 &&
                top[3] > 0 &&
                [top[0] + 7, top[0] + top[2] - 8].flatMap((bx) => [top[1] + 7, top[1] + top[3] - 8].map((by) => bolt(bx, by, Math.min(top[2], top[3]) > 40 ? 4 : 2)))}
            </g>
          );
        })}
      </>
    );
  },

  tank: (color, _theme, radius, aim) => machine(color, radius, aim),

  wreck: (_color, _theme, radius, aim) => machine(WRECK, radius, aim),

  shot(color, theme, radius, bounced) {
    return (
      <>
        <circle r={radius * 1.75} fill={color} opacity={0.3} />
        <circle r={radius * 1.25} fill={color} />
        <circle r={radius * 1.25} fill={SHADOW} opacity={0.5} />
        <circle r={radius} fill={bounced ? theme.text : color} />
      </>
    );
  },
};
