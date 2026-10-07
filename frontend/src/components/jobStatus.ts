import type { JobStatus } from "../api";
import type { Tone } from "./tones";

export const JOB_STATUS: Record<JobStatus, { tone: Tone; label: string }> = {
  pending: { tone: "neutral", label: "Pending" },
  running: { tone: "neutral", label: "Running" },
  completed: { tone: "good", label: "Completed" },
  failed: { tone: "critical", label: "Failed" },
};

export const ATTEMPT_OUTCOME: Record<string, { tone: Tone; label: string }> = {
  running: { tone: "neutral", label: "Running" },
  completed: { tone: "good", label: "Completed" },
  failed: { tone: "critical", label: "Failed" },
  lease_expired: { tone: "warning", label: "Worker died (lease expired)" },
  released: { tone: "warning", label: "Released on shutdown" },
};

export function jobDuration(started: string | null, ended: string | null): number | null {
  if (!started || !ended) return null;
  return (new Date(ended).getTime() - new Date(started).getTime()) / 1000;
}
