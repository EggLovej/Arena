// Neon Circuit's style (display/styles/neon.py): a midnight court under a faint
// grid, violet-edged walls, ringed robots whose barrels follow their aim, and
// glowing shots that turn white once they have bounced.

import type { Style } from "./RicochetBoard.tsx";

const GRID = 40;

export const neon: Style = {
  arena({ width, height, walls }, theme, compact, id) {
    return (
      <>
        <defs>
          <pattern id={`${id}-grid`} width={GRID} height={GRID} patternUnits="userSpaceOnUse">
            <path d={`M ${GRID} 0 L 0 0 0 ${GRID}`} fill="none" stroke={theme.inactive} strokeWidth={1} opacity={0.55} />
          </pattern>
          <radialGradient id={`${id}-floor`} cx="50%" cy="50%" r="70%">
            <stop offset="0%" stopColor={theme.accent} stopOpacity={0.08} />
            <stop offset="100%" stopColor={theme.accent} stopOpacity={0} />
          </radialGradient>
        </defs>

        <rect width={width} height={height} fill={theme.background} />
        <rect width={width} height={height} fill={`url(#${id}-grid)`} />
        <rect width={width} height={height} fill={`url(#${id}-floor)`} />

        {/* The arena's edge, which shots bounce off like any wall. */}
        <rect x={3} y={3} width={width - 6} height={height - 6} fill="none" stroke={theme.accent} strokeWidth={10} opacity={0.18} />
        <rect x={2} y={2} width={width - 4} height={height - 4} fill="none" stroke={theme.accent} strokeWidth={compact ? 6 : 3} />

        {walls.map(([x, y, w, h], index) => (
          <g key={index}>
            <rect x={x - 3} y={y - 3} width={w + 6} height={h + 6} rx={3} fill="none" stroke={theme.accent} strokeWidth={6} opacity={0.22} />
            <rect x={x} y={y} width={w} height={h} fill={theme.panel} stroke={theme.accent} strokeWidth={compact ? 5 : 2.5} />
            {!compact && w > 24 && h > 24 && (
              <rect x={x + 6} y={y + 6} width={w - 12} height={h - 12} fill="none" stroke={theme.accent} strokeWidth={1} opacity={0.35} />
            )}
          </g>
        ))}
      </>
    );
  },

  tank(color, theme, radius, aim) {
    return (
      <>
        <circle r={radius * 1.7} fill={color} opacity={0.14} />
        <rect transform={`rotate(${aim})`} x={radius * 0.2} y={-3.5} width={radius * 1.5} height={7} rx={2} fill={color} />
        <circle r={radius} fill={theme.background} stroke={color} strokeWidth={3} />
        <circle r={radius * 0.62} fill="none" stroke={color} strokeWidth={2.5} strokeDasharray="7 4" />
        <circle r={radius * 0.28} fill={color} />
      </>
    );
  },

  wreck(color, theme, radius) {
    const arm = radius * 0.55;
    return (
      <g opacity={0.75}>
        <circle r={radius} fill={theme.panel} stroke={theme.inactive} strokeWidth={3} />
        <path d={`M ${-arm} ${-arm} L ${arm} ${arm} M ${arm} ${-arm} L ${-arm} ${arm}`} stroke={color} strokeWidth={3} strokeLinecap="round" />
      </g>
    );
  },

  shot(color, theme, radius, bounced) {
    return (
      <>
        <circle r={radius * 2.4} fill={color} opacity={0.22} />
        <circle r={radius} fill={bounced ? theme.text : color} stroke={color} strokeWidth={1.5} />
      </>
    );
  },
};
