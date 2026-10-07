const integer = new Intl.NumberFormat("en-US");

export const formatCount = (value: number) => integer.format(value);

export const formatPercent = (ratio: number) =>
  `${(ratio * 100).toFixed(ratio >= 0.9995 || ratio === 0 ? 0 : 1)}%`;

export const formatUsd = (value: number) =>
  value < 0.01 && value > 0
    ? `$${value.toFixed(4)}`
    : new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
      }).format(value);

export function formatSeconds(seconds: number): string {
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 90) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  return `${(seconds / 60).toFixed(1)} min`;
}

// A bucket's start, labelled for its size: hours for short ranges, days
// for long ones. In the viewer's time zone (the API sends UTC).
export function formatBucket(iso: string, bucketSeconds: number): string {
  const date = new Date(iso);

  return bucketSeconds >= 86_400
    ? date.toLocaleDateString(undefined, { month: "short", day: "numeric" })
    : date.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "numeric",
      });
}

export function formatAxisTick(iso: string, bucketSeconds: number): string {
  const date = new Date(iso);

  return bucketSeconds >= 86_400 || bucketSeconds >= 6 * 3600
    ? date.toLocaleDateString(undefined, { month: "short", day: "numeric" })
    : date.toLocaleTimeString(undefined, { hour: "numeric" });
}
