import type { RangeName } from "../api";

const OPTIONS: { value: RangeName; label: string }[] = [
  { value: "24h", label: "Last 24 hours" },
  { value: "7d", label: "Last 7 days" },
  { value: "30d", label: "Last 30 days" },
];

// One filter row above everything it scopes.
export function RangeFilter({
  value,
  onChange,
}: {
  value: RangeName;
  onChange: (range: RangeName) => void;
}) {
  return (
    <div className="range-filter" role="group" aria-label="Time range">
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
