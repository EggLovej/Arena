// The speed menu under a replay or a match, in multiples of game time.

export function SpeedSelect({ speeds, value, onChange }: { speeds: number[]; value: number; onChange: (speed: number) => void }) {
  return (
    <label className="replay-speed">
      Speed
      <select value={value} onChange={(event) => onChange(Number(event.target.value))}>
        {speeds.map((speed) => (
          <option key={speed} value={speed}>
            {speed}×
          </option>
        ))}
      </select>
    </label>
  );
}
