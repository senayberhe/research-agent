import type { Tone } from "./tones";

// A status is never color alone: an icon and a label.

const ICON: Record<Tone, string> = {
  good: "●",
  warning: "▲",
  critical: "✕",
  neutral: "○",
};

export function StatusMark({ tone, label }: { tone: Tone; label: string }) {
  return (
    <>
      <span
        className="status-icon"
        style={{ color: `var(--${tone})` }}
        aria-hidden="true"
      >
        {ICON[tone]}
      </span>
      <span>{label}</span>
    </>
  );
}
