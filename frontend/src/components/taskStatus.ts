import type { RunStatus, TaskStatus } from "../api";
import type { Tone } from "./tones";

export const TASK_STATUS: Record<TaskStatus, { tone: Tone; label: string }> = {
  pending: { tone: "neutral", label: "Queued" },
  planning: { tone: "neutral", label: "Planning" },
  researching: { tone: "neutral", label: "Researching" },
  synthesizing: { tone: "neutral", label: "Writing summary" },
  in_progress: { tone: "neutral", label: "In progress" },
  completed: { tone: "good", label: "Completed" },
  failed: { tone: "critical", label: "Failed" },
};

// What the agent is doing right now.
export const RUN_PHASE: Record<RunStatus, string> = {
  created: "Starting",
  running: "Thinking",
  waiting_for_tool: "Searching",
  processing_result: "Reading results",
  completed: "Done",
  failed: "Stopped",
};

const TOOL_NAMES: Record<string, string> = {
  tavily: "Tavily",
  arxiv: "arXiv",
  wikipedia: "Wikipedia",
};

export const toolName = (tool: unknown) =>
  typeof tool === "string" ? (TOOL_NAMES[tool] ?? tool) : "a tool";
