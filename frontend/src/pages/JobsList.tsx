import { Link, useSearchParams } from "react-router-dom";

import { api, type JobStatus } from "../api";
import { Card } from "../components/Card";
import { JOB_STATUS, jobDuration } from "../components/jobStatus";
import { StatusMark } from "../components/status";
import { formatSeconds } from "../format";
import { useJobEvents } from "../notificationModel";
import { usePolling } from "../usePolling";

const FILTERS: { value: JobStatus | null; label: string }[] = [
  { value: null, label: "All" },
  { value: "running", label: "Running" },
  { value: "pending", label: "Pending" },
  { value: "completed", label: "Completed" },
  { value: "failed", label: "Failed" },
];

const PAGE_SIZE = 20;

export function JobsList() {
  // Filter and page in the URL, so links and refreshes keep them.
  const [params, setParams] = useSearchParams();
  const requested = params.get("status");
  const status = FILTERS.some((f) => f.value === requested) ? (requested as JobStatus) : null;
  const page = Math.max(0, Number(params.get("page") ?? 0) || 0);

  const jobs = usePolling(
    (signal) => api.listJobs(status, page * PAGE_SIZE, signal),
    `jobs-${status}-${page}`,
    5_000,
  );

  // A job finished, failed or was retried: refresh now.
  useJobEvents(() => true, jobs.reload);

  const setFilter = (value: JobStatus | null) =>
    setParams(value ? { status: value } : {}, { replace: true });

  const setPage = (value: number) =>
    setParams({ ...(status ? { status } : {}), ...(value ? { page: String(value) } : {}) });

  const data = jobs.data;
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <>
      <div className="page-header">
        <h1>Jobs</h1>
        <div className="range-filter" role="group" aria-label="Status">
          {FILTERS.map((filter) => (
            <button
              key={filter.label}
              type="button"
              aria-pressed={status === filter.value}
              onClick={() => setFilter(filter.value)}
            >
              {filter.label}
            </button>
          ))}
        </div>
      </div>

      {jobs.error && !data && (
        <div className="error-banner" role="alert">
          Couldn’t load jobs ({jobs.error.message}).
        </div>
      )}

      <Card title="Jobs Explorer" subtitle={data ? `${data.total} job${data.total === 1 ? "" : "s"}` : undefined}>
        {data && data.items.length === 0 ? (
          <p className="empty-note">No {status ?? ""} jobs.</p>
        ) : (
          <div className="table-wrap">
            <table className="jobs-table">
              <thead>
                <tr>
                  <th scope="col">Job</th>
                  <th scope="col">Question</th>
                  <th scope="col">Status</th>
                  <th scope="col">Attempts</th>
                  <th scope="col">Worker</th>
                  <th scope="col">Duration</th>
                  <th scope="col">Created</th>
                </tr>
              </thead>
              <tbody>
                {data?.items.map((job) => {
                  const state = JOB_STATUS[job.status];
                  const duration = jobDuration(job.started_at, job.completed_at);

                  return (
                    <tr key={job.id} className={job.status === "failed" ? "row-failed" : undefined}>
                      <td>
                        <Link to={`/jobs/${job.id}`}>#{job.id}</Link>
                      </td>
                      <td className="cell-question">
                        <Link to={`/jobs/${job.id}`}>{job.question}</Link>
                      </td>
                      <td>
                        <span className="task-status">
                          <StatusMark tone={state.tone} label={state.label} />
                        </span>
                      </td>
                      <td>
                        {job.attempts}
                        {job.attempts > 1 && <span className="tag">retried</span>}
                      </td>
                      <td className="cell-muted">{job.worker_id ?? "–"}</td>
                      <td className="cell-number">{duration !== null ? formatSeconds(duration) : "–"}</td>
                      <td className="cell-muted">
                        {new Date(job.created_at).toLocaleString(undefined, {
                          month: "short",
                          day: "numeric",
                          hour: "numeric",
                          minute: "2-digit",
                        })}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {data && pages > 1 && (
          <div className="pager">
            <button type="button" className="link-button" disabled={page === 0} onClick={() => setPage(page - 1)}>
              ← Newer
            </button>
            <span className="cell-muted">
              Page {page + 1} of {pages}
            </span>
            <button type="button" className="link-button" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>
              Older →
            </button>
          </div>
        )}
      </Card>
    </>
  );
}
