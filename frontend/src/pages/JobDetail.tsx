import { Link, useParams } from "react-router-dom";

import { api } from "../api";
import { Card } from "../components/Card";
import { ATTEMPT_OUTCOME, JOB_STATUS, jobDuration } from "../components/jobStatus";
import { StatusMark } from "../components/status";
import { formatSeconds } from "../format";
import { useJobEvents } from "../notificationModel";
import { usePolling } from "../usePolling";

const time = (iso: string | null) =>
  iso
    ? new Date(iso).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
      })
    : "–";

export function JobDetail() {
  const jobId = Number(useParams().jobId);

  const job = usePolling((signal) => api.job(jobId, signal), `job-${jobId}`, 3_000);
  const taskId = job.data?.task_id;
  const task = usePolling(
    (signal) => (taskId ? api.task(taskId, signal) : Promise.resolve(undefined)),
    `job-task-${taskId}`,
    0,
  );

  // This job changed: refresh now.
  useJobEvents((event) => event.job_id === jobId, job.reload);

  if (!Number.isInteger(jobId)) {
    return <p className="empty-note">Not a job.</p>;
  }

  const data = job.data;

  if (!data) {
    return job.error ? (
      <div className="error-banner" role="alert">
        Couldn’t load job #{jobId} ({job.error.message}).
      </div>
    ) : (
      <p className="empty-note">Loading…</p>
    );
  }

  const state = JOB_STATUS[data.status];
  const duration = jobDuration(data.started_at, data.completed_at);

  return (
    <>
      <Link to="/jobs" className="back-link">
        ← Jobs
      </Link>

      <div className="task-header">
        <h1>Job #{data.id}</h1>
        <div className="task-meta">
          <span className="task-status">
            <StatusMark tone={state.tone} label={state.label} />
          </span>
          {task.data && (
            <Link to={`/research/${data.task_id}`} className="cell-muted">
              {task.data.question}
            </Link>
          )}
        </div>
      </div>

      {data.status === "failed" && (
        <div className="error-banner failure-banner" role="alert">
          <strong>This job failed</strong>
          {data.attempts > 1 && ` after ${data.attempts} attempts`}
          {data.error ? `: ${data.error}` : "."}
          <div className="resume-row">
            <Link to={`/research/${data.task_id}`} className="primary-button">
              Open the research task
            </Link>
            <span className="cell-muted">Resume it from there if it has a checkpoint.</span>
          </div>
        </div>
      )}

      <div className="grid live-tiles">
        <section className="card">
          <p className="tile-label">Attempts</p>
          <p className="tile-value">{data.attempts}</p>
          {data.attempts > 1 && <p className="tile-note">{data.attempts - 1} retr{data.attempts === 2 ? "y" : "ies"}</p>}
        </section>
        <section className="card">
          <p className="tile-label">Worker</p>
          <p className="tile-value tile-value-small">{data.worker_id ?? "–"}</p>
        </section>
        <section className="card">
          <p className="tile-label">Duration</p>
          <p className="tile-value">{duration !== null ? formatSeconds(duration) : "–"}</p>
          <p className="tile-note">last attempt</p>
        </section>
        <section className="card">
          <p className="tile-label">Created</p>
          <p className="tile-value tile-value-small">{time(data.created_at)}</p>
        </section>
        <section className="card">
          <p className="tile-label">Started</p>
          <p className="tile-value tile-value-small">{time(data.started_at)}</p>
        </section>
        <section className="card">
          <p className="tile-label">Finished</p>
          <p className="tile-value tile-value-small">{time(data.completed_at)}</p>
        </section>
      </div>

      <div className="grid task-columns">
        <Card title="Attempts" subtitle="Every run of this job, including retries">
          {data.attempt_history.length === 0 ? (
            <p className="empty-note">No worker has started this job yet.</p>
          ) : (
            <div className="table-wrap">
              <table className="jobs-table">
                <thead>
                  <tr>
                    <th scope="col">#</th>
                    <th scope="col">Outcome</th>
                    <th scope="col">Worker</th>
                    <th scope="col">Started</th>
                    <th scope="col">Duration</th>
                    <th scope="col">Error</th>
                  </tr>
                </thead>
                <tbody>
                  {data.attempt_history.map((attempt) => {
                    const outcome = ATTEMPT_OUTCOME[attempt.outcome] ?? {
                      tone: "neutral" as const,
                      label: attempt.outcome,
                    };

                    return (
                      <tr key={attempt.attempt_number} className={outcome.tone === "critical" ? "row-failed" : undefined}>
                        <td>{attempt.attempt_number}</td>
                        <td>
                          <span className="task-status">
                            <StatusMark tone={outcome.tone} label={outcome.label} />
                          </span>
                        </td>
                        <td className="cell-muted">{attempt.worker_id ?? "–"}</td>
                        <td className="cell-muted">{time(attempt.started_at)}</td>
                        <td className="cell-number">
                          {attempt.duration_seconds !== null ? formatSeconds(attempt.duration_seconds) : "running"}
                        </td>
                        <td className="cell-error">{attempt.error ?? ""}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title="Events" subtitle="Newest first">
          <ol className="activity">
            {[...data.events].reverse().map((event) => (
              <li key={event.id}>
                <time dateTime={event.created_at}>
                  {new Date(event.created_at).toLocaleTimeString(undefined, {
                    hour: "numeric",
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </time>
                <span className="activity-text">
                  <strong>{event.event_type.replace(/_/g, " ")}</strong>
                  {event.summary ? ` · ${event.summary}` : ""}
                </span>
              </li>
            ))}
          </ol>
        </Card>
      </div>
    </>
  );
}
