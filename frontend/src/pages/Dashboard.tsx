import { useSearchParams } from "react-router-dom";

import { api, type RangeName } from "../api";
import { ChartCard, StatTile } from "../components/Card";
import {
  LatencyChart,
  ThroughputChart,
  TimeseriesTable,
} from "../components/charts";
import { ActiveJobs, WorkerStatus } from "../components/Panels";
import { RangeFilter } from "../components/RangeFilter";
import { formatCount, formatPercent, formatUsd } from "../format";
import { usePolling } from "../usePolling";

const RANGE_LABEL: Record<RangeName, string> = {
  "24h": "last 24 hours",
  "7d": "last 7 days",
  "30d": "last 30 days",
};

export function Dashboard() {
  // In the URL (?range=7d), so a link or a refresh keeps the view.
  const [params, setParams] = useSearchParams();
  const requested = params.get("range");
  const range: RangeName =
    requested === "7d" || requested === "30d" ? requested : "24h";
  const setRange = (value: RangeName) =>
    setParams(value === "24h" ? {} : { range: value }, { replace: true });

  const metrics = usePolling(
    (signal) => api.systemMetrics(range, signal),
    `metrics-${range}`,
  );
  const series = usePolling(
    (signal) => api.jobTimeseries(range, signal),
    `series-${range}`,
  );

  // The job latency SLO target, from the API (app/core/slo.py), so the
  // chart's SLO line always matches what the backend measures against.
  const slo = usePolling((signal) => api.slo(signal), "slo", 60_000);
  const latencyTarget = slo.data?.job_latency.target_seconds;

  const buckets = series.data?.buckets ?? [];
  const bucketSeconds = series.data?.bucket_seconds ?? 3600;

  // Jobs that finished in the range, as the charts count them.
  const completed = buckets.reduce((sum, b) => sum + b.completed, 0);
  const failed = buckets.reduce((sum, b) => sum + b.failed, 0);
  const finished = completed + failed;

  const agents = metrics.data?.agents;
  const error = metrics.error ?? series.error;
  const refreshing = metrics.loading || series.loading;

  return (
    <>
      <div className="page-header">
        <h1>Research Overview</h1>
        <RangeFilter value={range} onChange={setRange} />
      </div>

      {error && (
        <div className="error-banner" role="alert">
          Couldn’t load the latest data ({error.message}). Showing the last
          numbers received.
        </div>
      )}

      <div className={refreshing && series.data ? "refreshing" : undefined}>
        <div className="grid tiles">
          <StatTile
            label="Jobs"
            value={series.data ? formatCount(finished) : null}
            note={
              series.data
                ? `finished in the ${RANGE_LABEL[range]}${failed ? ` · ${formatCount(failed)} failed` : ""}`
                : undefined
            }
          />
          <StatTile
            label="Success"
            // No finished jobs: no data, not 100%.
            value={finished ? formatPercent(completed / finished) : null}
            note={finished ? `${formatCount(completed)} of ${formatCount(finished)} completed` : "no jobs finished in this period"}
          />
          <StatTile
            label="Cost"
            value={agents ? formatUsd(agents.estimated_cost_usd) : null}
            note={
              agents
                ? agents.unpriced_runs
                  ? `estimated · ${agents.unpriced_runs} run(s) with unknown pricing not included`
                  : `estimated LLM cost · ${formatCount(agents.total_tokens)} tokens`
                : undefined
            }
          />
        </div>

        <div className="grid charts">
          <ChartCard
            title="Job Throughput"
            subtitle="Jobs finished per period"
            chart={
              finished ? (
                <ThroughputChart buckets={buckets} bucketSeconds={bucketSeconds} />
              ) : (
                <div className="chart-empty">No jobs finished in this period</div>
              )
            }
            table={<TimeseriesTable buckets={buckets} bucketSeconds={bucketSeconds} />}
          />
          <ChartCard
            title="Job Latency (p95)"
            subtitle={
              latencyTarget !== undefined
                ? `95th percentile run time, SLO ${latencyTarget}s`
                : "95th percentile run time"
            }
            chart={
              finished ? (
                <LatencyChart
                  buckets={buckets}
                  bucketSeconds={bucketSeconds}
                  targetSeconds={latencyTarget}
                />
              ) : (
                <div className="chart-empty">No jobs finished in this period</div>
              )
            }
            table={<TimeseriesTable buckets={buckets} bucketSeconds={bucketSeconds} />}
          />
        </div>

        {metrics.data && (
          <div className="grid panels">
            <ActiveJobs queue={metrics.data.queue} />
            <WorkerStatus workers={metrics.data.workers} rangeLabel={RANGE_LABEL[range]} />
          </div>
        )}
      </div>
    </>
  );
}
