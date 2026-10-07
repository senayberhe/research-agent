import { api, type SLOEntry, type SLOStatus } from "../api";
import { Card } from "../components/Card";
import { StatusMark } from "../components/status";
import type { Tone } from "../components/tones";
import { formatSeconds } from "../format";
import { usePolling } from "../usePolling";

const STATUS: Record<SLOStatus, { tone: Tone; label: string }> = {
  healthy: { tone: "good", label: "Healthy" },
  breached: { tone: "critical", label: "Breached" },
  no_data: { tone: "neutral", label: "No data" },
  unknown: { tone: "neutral", label: "Measured by Prometheus" },
};

const pct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? "–" : `${(value * 100).toFixed(digits)}%`;

const ms = (value: number | null | undefined) =>
  value === null || value === undefined ? "–" : formatSeconds(value / 1000);

function Budget({ entry }: { entry: SLOEntry }) {
  const budget = entry.error_budget;

  if (!budget) return <p className="tile-note">No error budget</p>;
  if (budget.budget_remaining === null) return <p className="tile-note">Error budget: no data</p>;

  const remaining = budget.budget_remaining;
  const tone: Tone = budget.exhausted ? "critical" : remaining < 0.25 ? "warning" : "good";

  return (
    <div className="budget">
      <div className="budget-label">
        <span>Error budget left</span>
        <strong>{pct(remaining, 0)}</strong>
      </div>
      <div className="budget-track" role="img" aria-label={`Error budget ${pct(remaining, 0)} left`}>
        <div className="budget-fill" style={{ width: `${Math.max(0, Math.min(1, remaining)) * 100}%`, background: `var(--${tone})` }} />
      </div>
    </div>
  );
}

function SLOCard({
  title,
  description,
  entry,
  actual,
}: {
  title: string;
  description: string;
  entry: SLOEntry;
  actual: string;
}) {
  const status = STATUS[entry.status];

  return (
    <section className="card slo-card">
      <div className="slo-head">
        <h2 className="card-title">{title}</h2>
        <span className="task-status">
          <StatusMark tone={status.tone} label={status.label} />
        </span>
      </div>
      <p className="card-subtitle">{description}</p>
      <p className="tile-value">{actual}</p>
      <Budget entry={entry} />
    </section>
  );
}

export function SLO() {
  const report = usePolling((signal) => api.slo(signal), "slo-page", 30_000);
  const data = report.data;

  return (
    <>
      <div className="page-header">
        <h1>Service level objectives</h1>
        {data && (
          <span className="updated">
            Last {Math.round(data.window_days)} days ·{" "}
            {new Date(data.window_start).toLocaleDateString()} – {new Date(data.window_end).toLocaleDateString()}
          </span>
        )}
      </div>

      {report.error && !data && (
        <div className="error-banner" role="alert">
          Couldn’t load SLOs ({report.error.message}).
        </div>
      )}

      {data && (
        <>
          <div className="grid slo-grid">
            <SLOCard
              title="Research job success"
              description={`Target: ${pct(data.research_job_success.target)} of finished jobs complete`}
              entry={data.research_job_success}
              actual={pct(data.research_job_success.actual, 2)}
            />
            <SLOCard
              title="Tool success"
              description={`Target: ${pct(data.tool_success.target)} of tool calls succeed`}
              entry={data.tool_success}
              actual={pct(data.tool_success.actual, 2)}
            />
            <SLOCard
              title="Job latency (p95)"
              description={`Target: p95 under ${data.job_latency.target_seconds}s`}
              entry={data.job_latency}
              actual={ms(data.job_latency.actual_ms)}
            />
            <SLOCard
              title="API availability"
              description={`Target: ${pct(data.api_availability.target)} (up{job="research-api"} in Prometheus)`}
              entry={data.api_availability}
              actual="See Grafana"
            />
          </div>

          <Card
            title="Tool latency (p95), per tool"
            subtitle={`Each tool against the ${data.tool_latency.target}s target`}
          >
            {Object.keys(data.tool_latency.by_tool ?? {}).length === 0 ? (
              <p className="empty-note">No tool calls in this window.</p>
            ) : (
              <div className="table-wrap">
                <table className="jobs-table">
                  <thead>
                    <tr>
                      <th scope="col">Tool</th>
                      <th scope="col">Status</th>
                      <th scope="col">p50</th>
                      <th scope="col">p95</th>
                      <th scope="col">p99</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.tool_latency.by_tool ?? {}).map(([tool, latency]) => (
                      <tr key={tool}>
                        <td>{tool}</td>
                        <td>
                          <span className="task-status">
                            <StatusMark tone={STATUS[latency.status].tone} label={STATUS[latency.status].label} />
                          </span>
                        </td>
                        <td className="cell-number">{ms(latency.p50_ms)}</td>
                        <td className="cell-number">{ms(latency.p95_ms)}</td>
                        <td className="cell-number">{ms(latency.p99_ms)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}
    </>
  );
}
