// A small robot portrait generated from a bot's id, so every bot in every game
// gets a face without anyone drawing one. The same id always gives the same face.

function hash(text: string): number {
  let value = 2166136261;
  for (let index = 0; index < text.length; index++) {
    value ^= text.charCodeAt(index);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}

export function BotAvatar({ id, kind, size = 40 }: { id: string; kind?: string; size?: number }) {
  const seed = hash(id);
  const pick = (count: number, shift: number) => (seed >>> shift) % count;
  const hue = seed % 360;
  const background = `hsl(${hue} 55% 88%)`;
  const face = `hsl(${hue} 45% 42%)`;
  const dark = `hsl(${hue} 50% 22%)`;
  const eye = pick(3, 8);
  const mouth = pick(3, 12);
  const antenna = pick(3, 16);
  const neural = kind === "neural";

  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true" className="bot-avatar">
      <rect width="40" height="40" rx="10" fill={background} />
      {antenna === 0 && <line x1="20" y1="4" x2="20" y2="10" stroke={dark} strokeWidth="2" />}
      {antenna === 0 && <circle cx="20" cy="4" r="2.5" fill={dark} />}
      {antenna === 1 && <path d="M13 10 L10 4 M27 10 L30 4" stroke={dark} strokeWidth="2" strokeLinecap="round" />}
      <rect x="8" y="10" width="24" height="22" rx={antenna === 2 ? 11 : 6} fill={face} />
      {eye === 0 && (
        <>
          <circle cx="15" cy="19" r="3" fill="#fff" />
          <circle cx="25" cy="19" r="3" fill="#fff" />
        </>
      )}
      {eye === 1 && (
        <>
          <rect x="11.5" y="17" width="7" height="4" rx="1.5" fill="#fff" />
          <rect x="21.5" y="17" width="7" height="4" rx="1.5" fill="#fff" />
        </>
      )}
      {eye === 2 && <rect x="11" y="16.5" width="18" height="5" rx="2.5" fill="#fff" />}
      {mouth === 0 && <rect x="15" y="25" width="10" height="2.5" rx="1.25" fill={dark} />}
      {mouth === 1 && <path d="M14 25 Q20 29 26 25" stroke={dark} strokeWidth="2" fill="none" strokeLinecap="round" />}
      {mouth === 2 && <path d="M14 26 h2 v-1.5 h2 v1.5 h2 v-1.5 h2 v1.5 h2" stroke={dark} strokeWidth="1.5" fill="none" />}
      {neural && (
        <g stroke="#fff" strokeWidth="1" opacity="0.7">
          <circle cx="33" cy="33" r="5.5" fill={dark} />
          <path d="M30.5 33 h5 M33 30.5 v5 M31.2 31.2 l3.6 3.6 M34.8 31.2 l-3.6 3.6" />
        </g>
      )}
    </svg>
  );
}
