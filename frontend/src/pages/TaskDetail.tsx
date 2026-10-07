import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Link, useParams } from "react-router-dom";
import remarkGfm from "remark-gfm";

import { api, type TaskProgress, type TimelineEvent } from "../api";
import { Card, StatTile } from "../components/Card";
import { StatusMark } from "../components/status";
import { ToolCalls } from "../components/ToolCalls";
import { RUN_PHASE, TASK_STATUS, toolName } from "../components/taskStatus";
import { formatCount, formatSeconds, formatUsd } from "../format";
import { useJobEvents } from "../notificationModel";
import { usePolling } from "../usePolling";

// How often to refresh while the task is going.
const LIVE_INTERVAL_MS = 2_000;

// One line per event worth showing (tool searches come from the job
// events; the matching research_step rows would repeat them).
function describe(event: TimelineEvent): string | null {
  const m = event.metadata;

  switch (event.event_type) {
    case "created":
      return "Queued";
    case "claimed":
      return `Picked up by ${m.worker_id ?? "a worker"}${Number(m.attempt) > 1 ? ` (attempt ${m.attempt})` : ""}`;
    case "agent_started":
      return "Agent started";
    case "tool_started":
      return `Searching ${toolName(m.tool)}${m.query ? `: “${m.query}”` : ""}`;
    case "tool_completed":
      return `${toolName(m.tool)} returned${typeof m.duration_ms === "number" ? ` in ${formatSeconds(m.duration_ms / 1000)}` : ""}`;
    case "tool_failed":
      return `${toolName(m.tool)} failed${m.error ? `: ${m.error}` : ""}`;
    case "agent_completed":
      return "Agent finished";
    case "agent_failed":
      return event.message ?? "Agent failed";
    case "completed":
      return "Research completed";
    case "failed":
      return `Failed${event.message ? `: ${event.message}` : ""}`;
    case "requeued":
      return "Queued again";
    case "lease_expired":
      return "Worker stopped responding; job returned to the queue";
    case "released":
      return "Worker shut down; job returned to the queue";
    default:
      return null;
  }
}

function eventTone(type: string) {
  if (type === "completed" || type === "agent_completed") return "good" as const;
  if (type.endsWith("failed") || type === "lease_expired") return "critical" as const;
  return "neutral" as const;
}

function Activity({ events }: { events: TimelineEvent[] }) {
  const shown = events
    .filter((event) => event.source !== "step")
    .map((event) => ({ event, text: describe(event) }))
    .filter((item): item is { event: TimelineEvent; text: string } => item.text !== null)
    // Newest first: what's happening now is at the top.
    .reverse();

  if (shown.length === 0) {
    return <p className="empty-note">Nothing has happened yet.</p>;
  }

  return (
    <ol className="activity">
      {shown.map(({ event, text }, index) => (
        <li key={`${event.timestamp}-${event.event_type}-${index}`}>
          <time dateTime={event.timestamp}>
            {new Date(event.timestamp).toLocaleTimeString(undefined, {
              hour: "numeric",
              minute: "2-digit",
              second: "2-digit",
            })}
          </time>
          <span className="activity-text">
            <StatusMark tone={eventTone(event.event_type)} label={text} />
          </span>
        </li>
      ))}
    </ol>
  );
}

function LiveTiles({ progress }: { progress: TaskProgress }) {
  const run = progress.run;

  return (
    <div className="grid live-tiles">
      <StatTile
        label="Phase"
        value={run ? RUN_PHASE[run.status] : TASK_STATUS[progress.task.status].label}
      />
      <StatTile label="Iterations" value={run ? formatCount(run.iterations) : null} note="LLM calls" />
      <StatTile
        label="Tool calls"
        // The checkpoint's count lags by the searches running now.
        value={run ? formatCount(Math.max(run.tool_calls, progress.tool_calls.length)) : null}
      />
      <StatTile
        label="Tokens"
        value={run ? formatCount(run.total_tokens) : null}
        note={run ? `${formatCount(run.input_tokens)} in · ${formatCount(run.output_tokens)} out` : undefined}
      />
      <StatTile
        label="Cost"
        value={run ? (run.estimated_cost_usd !== null ? formatUsd(run.estimated_cost_usd) : "Unknown") : null}
        note={run && run.estimated_cost_usd === null ? "model pricing unknown" : "estimated"}
      />
      <StatTile label="Elapsed" value={run ? formatSeconds(run.elapsed_seconds) : null} />
    </div>
  );
}

export function TaskDetail() {
  const taskId = Number(useParams().taskId);
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState<string | null>(null);
  // Bumped after a resume, to start polling again.
  const [generation, setGeneration] = useState(0);

  // Polls every 2s until the task is finished; once finished, no more
  // requests (finishedKey changes the effect, interval 0 = fetch once).
  const [finished, setFinished] = useState(false);
  const progress = usePolling(
    async (signal) => {
      const data = await api.progress(taskId, signal);
      setFinished(data.finished);
      return data;
    },
    `progress-${taskId}-${generation}-${finished}`,
    finished ? 0 : LIVE_INTERVAL_MS,
  );

  // Something happened to this task (finished, failed, a tool failed):
  // refresh now rather than at the next poll.
  useJobEvents((event) => event.task_id === taskId, () => {
    setFinished(false);
    progress.reload();
  });

  if (!Number.isInteger(taskId)) {
    return <p className="empty-note">Not a research task.</p>;
  }

  const data = progress.data;

  const resume = async () => {
    setResuming(true);
    setResumeError(null);
    try {
      await api.resume(taskId);
      setFinished(false);
      setGeneration((value) => value + 1);
    } catch (failure) {
      setResumeError((failure as Error).message);
    } finally {
      setResuming(false);
    }
  };

  if (!data) {
    return progress.error ? (
      <div className="error-banner" role="alert">
        Couldn’t load this research task ({progress.error.message}).
      </div>
    ) : (
      <p className="empty-note">Loading…</p>
    );
  }

  const status = TASK_STATUS[data.task.status];
  const failure = data.task.status === "failed" ? (data.run?.error ?? data.job?.error) : null;

  return (
    <>
      <Link to="/research" className="back-link">
        ← Research
      </Link>

      <div className="task-header">
        <h1>{data.task.question}</h1>
        <div className="task-meta">
          <span className="task-status">
            <StatusMark tone={status.tone} label={status.label} />
          </span>
          {data.finished ? (
            <span className="updated">Finished</span>
          ) : (
            <span className="live-indicator" aria-live="polite">
              <span className="live-dot" aria-hidden="true" /> Live · updates every 2 s
            </span>
          )}
        </div>
      </div>

      {progress.error && (
        <div className="error-banner" role="alert">
          Lost contact with the API ({progress.error.message}); retrying.
        </div>
      )}

      {failure && (
        <div className="error-banner" role="alert">
          <strong>The research failed:</strong> {failure}
          {data.can_resume && (
            <div className="resume-row">
              <button type="button" className="primary-button" onClick={resume} disabled={resuming}>
                {resuming ? "Resuming…" : "Resume from last checkpoint"}
              </button>
              {resumeError && <span className="form-error">{resumeError}</span>}
            </div>
          )}
        </div>
      )}

      <LiveTiles progress={data} />

      <div className="grid task-columns">
        <Card title="Summary" subtitle={data.task.summary ? undefined : "Appears here when the research finishes"}>
          {data.task.summary ? (
            <div className="markdown">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a: ({ children, ...props }) => (
                    <a {...props} target="_blank" rel="noopener noreferrer">
                      {children}
                    </a>
                  ),
                }}
              >
                {data.task.summary}
              </ReactMarkdown>
            </div>
          ) : (
            <p className="empty-note">
              {data.finished ? "No summary was produced." : `${data.run ? RUN_PHASE[data.run.status] : status.label}…`}
            </p>
          )}
        </Card>

        <Card title="Activity" subtitle="Newest first">
          <Activity events={data.events} />
        </Card>
      </div>

      <div className="tool-calls-section">
        <ToolCalls calls={data.tool_calls} />
      </div>
    </>
  );
}
