import type { HealthStatus } from "../api";

export type Tone = "good" | "warning" | "critical" | "neutral";

export const HEALTH_TONE: Record<HealthStatus, Tone> = {
  healthy: "good",
  degraded: "warning",
  unhealthy: "critical",
};

export const HEALTH_LABEL: Record<HealthStatus, string> = {
  healthy: "System Healthy",
  degraded: "System Degraded",
  unhealthy: "System Unhealthy",
};
