import type { SystemMetrics, Worker } from "../api";
import { formatSeconds } from "../format";
import { Card } from "./Card";
import { StatusMark } from "./status";
import type { Tone } from "./tones";

export function ActiveJobs({ queue }: { queue: SystemMetrics["queue"] }) {
  return (
    <Card title="Active Jobs" subtitle="Now">
      <ul className="status-list">
        <li>
          <StatusMark tone="good" label="Running" />
          <span className="label" />
          <span className="value">{queue.running_jobs}</span>
        </li>
        <li>
          <StatusMark tone="neutral" label="Pending" />
          <span className="label">
            {queue.oldest_pending_seconds !== null && queue.pending_jobs > 0 && (
              <span className="detail">
                {" "}
                oldest waiting {formatSeconds(queue.oldest_pending_seconds)}
              </span>
            )}
          </span>
          <span className="value">{queue.pending_jobs}</span>
        </li>
        {queue.expired_leases > 0 && (
          <li>
            <StatusMark tone="critical" label="Stuck (lease expired)" />
            <span className="label" />
            <span className="value">{queue.expired_leases}</span>
          </li>
        )}
      </ul>
    </Card>
  );
}

const WORKER_STATE: Record<Worker["state"], { tone: Tone; label: string }> = {
  busy: { tone: "good", label: "healthy" },
  idle: { tone: "good", label: "healthy" },
  unresponsive: { tone: "critical", label: "unresponsive" },
};

export function WorkerStatus({
  workers,
  rangeLabel,
}: {
  workers: Worker[];
  rangeLabel: string;
}) {
  return (
    <Card title="Worker Status" subtitle={`Workers that ran a job in the ${rangeLabel}`}>
      {workers.length === 0 ? (
        <p className="empty-note">No worker has claimed a job in this period.</p>
      ) : (
        <ul className="status-list">
          {workers.map((worker) => {
            const state = WORKER_STATE[worker.state];

            return (
              <li key={worker.worker_id}>
                <span className="label">{worker.worker_id}</span>
                <span className="detail">
                  {worker.state === "busy"
                    ? `busy · job #${worker.current_job_id}`
                    : worker.state === "idle"
                      ? "idle"
                      : ""}
                </span>
                <StatusMark tone={state.tone} label={state.label} />
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}
